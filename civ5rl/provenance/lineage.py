"""Parent artifacts are inputs, not constraints on the child's implementation."""
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import read_json


def verify_parent_lineage(parent, previous, current, *, same_corpus=False):
    """Require an unchanged corpus only when the experiment declares it.

    Checkpoint bytes, architecture and optimizer compatibility belong to each
    loader. Source moves and a different environment never invalidate a parent.
    """
    if same_corpus and previous['corpus_hash']!=current['corpus_hash']:
        raise ValueError('Continuation corpus differs from parent')
    return []


def current_parent_lineage(parent, config, manifest, *, same_corpus=False):
    import tempfile
    previous=checkpoint.verify_frozen_passport(parent,sources=False)
    with tempfile.TemporaryDirectory() as directory:
        current=checkpoint.passport(directory,config,manifest)
    return current,verify_parent_lineage(parent,previous,current,same_corpus=same_corpus)


def verify_existing_run(out, config, current):
    identity=checkpoint.verify_frozen_passport(out,config=config,sources=False)
    if not checkpoint.same_execution_identity(identity,current):
        raise ValueError('Run/data identity mismatch')
    return identity
