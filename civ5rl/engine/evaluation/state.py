"""Resident per-battle GRU states and reusable inference batch storage."""

from __future__ import annotations

import torch


class _StateLease:
    """Return storage when either the executor or the owning agent releases it."""

    def __init__(self, bank, slot):
        self.bank, self.slot = bank, slot

    def release(self):
        if self.bank is not None:
            self.bank.free.append(self.slot)
            self.bank = None

    def __del__(self):
        self.release()


class RecurrentStates:
    """Stable state slots, packed/unpacked with fused foreach copies on CUDA.

    A member's hidden tensor borrows its slot until release/reset. Callers saving
    historical states must copy them, as source-prefix capture does. Slabs never
    move when capacity grows; one inference thread owns allocation and reuse.
    All tensor operations run on that thread's current CUDA stream. No device
    index uploads, host staging, or per-decision synchronization are needed.
    """

    def __init__(self, hidden_size, device, capacity=32):
        self.hidden_size, self.device = hidden_size, device
        self.capacity = max(1, capacity)
        self.slabs = []
        self.free = []
        self.batch = None
        self.batch_rows = []

    def _slot(self):
        if not self.free:
            slab = torch.empty(
                (1, self.capacity, self.hidden_size),
                device=self.device,
                dtype=torch.float32,
            )
            self.slabs.append(slab)
            self.free.extend(reversed(slab.split(1, dim=1)))
        return self.free.pop()

    def gather(self, agents):
        """Return contiguous h0 in request order, importing reset/prefill state."""
        if len({id(agent) for agent in agents}) != len(agents):
            raise ValueError("A battle cannot occur twice in one inference batch")
        states = []
        for agent in agents:
            lease = getattr(agent, "_resident_hidden", None)
            if lease is None or lease.bank is not self:
                slot = self._slot()
                if agent.hidden is None:
                    slot.zero_()
                else:
                    if agent.hidden.shape != slot.shape:
                        raise ValueError("Invalid imported recurrent state shape")
                    slot.copy_(agent.hidden)
                # Queue the import before releasing storage from another bank.
                self.release(agent)
                agent._resident_hidden = _StateLease(self, slot)
            else:
                slot = lease.slot
                if agent.hidden is not slot:
                    if agent.hidden is None:
                        slot.zero_()
                    else:
                        if agent.hidden.shape != slot.shape:
                            raise ValueError("Invalid imported recurrent state shape")
                        slot.copy_(agent.hidden)
            agent.hidden = slot
            states.append(slot)
        count = len(states)
        if len(self.batch_rows) < count:
            self.batch = torch.empty(
                (1, max(count, self.capacity), self.hidden_size),
                device=self.device,
                dtype=torch.float32,
            )
            self.batch_rows = list(self.batch.split(1, dim=1))
        torch._foreach_copy_(self.batch_rows[:count], states)
        return self.batch[:, :count]

    def scatter(self, agents, hidden):
        """Update only participating battles; other states retain their values."""
        torch._foreach_copy_([agent.hidden for agent in agents], list(hidden.split(1, dim=1)))

    @staticmethod
    def release(agent):
        """Return a member's slot; the next owner imports or zeroes its state."""
        lease = getattr(agent, "_resident_hidden", None)
        if lease is not None:
            lease.release()
            del agent._resident_hidden
        agent.hidden = None
