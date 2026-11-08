# Remote experiment deployments

The primary operational tools are `ctl_rl.py`, `remote_rl.py` and `bench_rl.py`.
`remote_rl.py` operates on an existing instance; renting, GPU selection and bans
remain in the Vast.ai skill.

```powershell
python tools/remote_rl.py package --config configs/rl-ppo-target-bc-t05.json --out DEPLOYMENT --extra-input CASE_CARRIER.pt
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode tune --instance INSTANCE_JSON
python tools/remote_rl.py pull --deployment DEPLOYMENT
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode run
python tools/remote_rl.py status --deployment DEPLOYMENT
python tools/remote_rl.py pull --deployment DEPLOYMENT --keep-instance
python tools/remote_rl.py launch --deployment DEPLOYMENT --mode resume
```

Package output must be empty. Source, inherited configs, checkpoint/manifest,
open maps, native build inputs and explicit extra inputs are fixed in archives
with size/SHA256/member fingerprints. A Git bundle preserves the base commit
without needing a network clone. No local scientific output or rollout is
implicitly uploaded. For after-target-BC tuning, pass exactly one extra `.pt`
case carrier containing the fixed cases required by the streaming tuner.
It is not a T=1 behavior-distribution reference.

`INSTANCE_JSON` requires `instance_id` (or `id`), `host` (or `SSH_host`) and `port`
(or `SSH_port`). Supply it at package or first launch. Only ID/endpoint are
retained; username and private key come from `ASSISTANT_REMOTE_CONN` or the SSH
skill's credential environment variables. A bound instance cannot change.
Remote root/project/venv paths derive from the package fingerprint under
`/workspace/civ5-runs/`; no experiment-specific IP, username or PID is embedded.

| Experiment kind | run | tune | resume |
|---|---|---|---|
| `rl01.ppo_after_target_bc` | yes | yes | yes |
| `rl01.ppo_target_scale` | yes | no | yes |
| `rl01.ppo_scale_up`, including continuation | yes | no | no |

Unsupported modes fail explicitly. `tune` invokes the existing streaming tuner
with its 600-second budget. After-target-BC `run` requires a confirmed comparable
result for this exact package/config. Pull the tuning job before launching run.

Execution uses the installed **remote-python-ssh** skill's durable runner,
detach/attach and verified post-sync. There is no second supervisor. A dedicated
payload directory is mirrored; the project, venv and scientific output remain
outside this destructive mirror. The worker prepares dependencies/native libs
once, records machine checks and invokes the existing training CLI. Deployment
metadata records runtime, durable job ID and operation receipts.
Repeating launch while a linked job is outstanding attaches to it. An uncertain
start is retained and recovered through its matching durable registry; the CLI
never blindly creates a replacement job. Unrelated jobs are not attached.

`resume` requires an existing remote run on the same instance, unchanged frozen
sources/inputs and previous model/Adam/RNG/budget. It does not bootstrap or
overlay modified local sources. Local Windows rollouts/evaluation caches are
not migrated. BC resume and new training loops are outside this interface.

`status` is bounded and read-only: status, recent updates and telemetry without
attach or signals. `pull` attaches to finalize the transport; if training is
active it waits without stopping it. The worker archives results after runner
completion, pause or technical error, publishing an operation token, byte size,
SHA256 and member hashes. Pull checks final execution/post-sync, descriptor
identity, ZIP safety and hashes before extraction. Incomplete/corrupt downloads
are not successful. Scientific audit is recorded separately from transport
integrity; failures retain downloaded evidence.

Instances are kept by default. Only explicit `pull --delete-instance` delegates
deletion of the bound instance ID to the installed **vastai-instance-manager**
script, after verified transport and mandatory integrity checks. Failed tuning,
missing mandatory checks or transfer errors preserve the instance. Repeating
pull accepts identical files; conflicts are rejected instead of overwritten.

Adapters are checked with substituted transport, without renting a machine,
training or evaluating battles. Service integration requires the installed
skills, configured credentials and a Linux C++ compiler.
