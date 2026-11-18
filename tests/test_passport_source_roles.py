"""Protect resume state without making parent artifacts depend on source layout."""
import hashlib
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from civ5rl.provenance import checkpoint,execution
from civ5rl.provenance.common import read_json,write_json

class PassportSourceRoleTests(unittest.TestCase):
    def git(self,root,*args):
        return subprocess.check_output(['git',*args],cwd=root,stderr=subprocess.STDOUT,text=True).strip()
    def commit(self,root):
        self.git(root,'add','.')
        self.git(root,'-c','user.name=Test','-c','user.email=test@example.com','commit','-qm','fixture')
    def test_resume_requires_committed_code_and_ignored_state_survives_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.git(root,'init','-q')
            code=root/'civ5rl/engine/fixture.py';code.parent.mkdir(parents=True)
            code.write_text('value=1\n');(root/'.gitignore').write_text('artifacts/\nbuild/\n')
            self.commit(root);old=self.git(root,'rev-parse','HEAD')
            run=root/'artifacts/run';run.mkdir(parents=True)
            import random
            import torch
            state=run/'checkpoint.pt'
            model=torch.nn.Linear(2,1);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            optimizer.zero_grad();model(torch.ones(1,2)).sum().backward();optimizer.step()
            model.variant='smoke'
            shuffle=random.Random(17);config={'lr':.01};identity={'run_id':'fixture'}
            checkpoint.save(state,model,optimizer,{'steps':1,'cursor':7},shuffle,config,identity)
            original=state.read_bytes();expected_draw=(torch.rand(1),shuffle.random())
            expected_model={k:v.detach().clone() for k,v in model.state_dict().items()}

            binary=root/'build/combat.dll';binary.parent.mkdir();binary.write_bytes(b'original build')
            with patch.object(execution,'ROOT',root):
                execution.execution_session(run,'prepare')
                self.assertFalse((run/'execution.json').exists())
                code.write_text('value=2\n')
                with self.assertRaisesRegex(ValueError,'Commit code'):execution.execution_session(run,'train')
                code.write_text('value=1\n');execution.execution_session(run,'train')
                receipt=(run/'execution.json').read_bytes()
                execution.execution_session(run,'resume')
                code.rename(code.with_name('renamed.py'));self.commit(root)
                with self.assertRaisesRegex(ValueError,'Resume requires Git commit'):
                    execution.execution_session(run,'resume')
                self.git(root,'checkout','--detach',old)
                execution.execution_session(run,'resume')
                self.assertEqual(state.read_bytes(),original)
                restored_model=torch.nn.Linear(2,1)
                restored_optimizer=torch.optim.Adam(restored_model.parameters(),lr=.01)
                restored=checkpoint.restore(state,restored_model,restored_optimizer,shuffle,config,identity)
                self.assertEqual(restored,{'steps':1,'cursor':7})
                for key,value in restored_model.state_dict().items():
                    torch.testing.assert_close(value,expected_model[key],atol=0,rtol=0)
                self.assertTrue(all(int(v['step'])==1 for v in restored_optimizer.state.values()))
                for key,row in optimizer.state_dict()['state'].items():
                    for field in ('exp_avg','exp_avg_sq'):
                        torch.testing.assert_close(row[field],restored_optimizer.state_dict()['state'][key][field],atol=0,rtol=0)
                torch.testing.assert_close(torch.rand(1),expected_draw[0],atol=0,rtol=0)
                self.assertEqual(shuffle.random(),expected_draw[1])
                self.assertEqual((run/'execution.json').read_bytes(),receipt)
                binary.write_bytes(b'other build')
                with self.assertRaisesRegex(ValueError,'compiled binaries'):execution.execution_session(run,'resume')
                binary.write_bytes((run/'execution-binaries/combat.dll').read_bytes())
                execution.execution_session(run,'resume')
    def test_parent_inputs_ignore_source_layout_but_reject_changed_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'
            with patch.object(checkpoint.subprocess,'check_output',return_value='old-commit\n'):
                checkpoint.passport(out,{'lr':1},{'cases':[1]})
            self.assertFalse((out/'sources.zip').exists())
            self.assertEqual(read_json(out/'passport.json')['source_hashes'],{})
            checkpoint.verify_execution_passport(out,{'lr':1},{'cases':[1]})
            with self.assertRaisesRegex(ValueError,'configuration'):
                checkpoint.verify_execution_passport(out,{'lr':2},{'cases':[1]})
            write_json(out/'corpus-manifest.json',{'cases':[2]})
            with self.assertRaisesRegex(ValueError,'corpus manifest'):checkpoint.verify_frozen_passport(out)
    def test_explicit_audit_can_still_verify_historical_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(checkpoint.subprocess,'check_output',return_value='commit\n'):
                meta=checkpoint.passport(out,{}, {})
            original=b'old producer code'
            meta['source_hashes']={'old/module.py':hashlib.sha256(original).hexdigest()}
            write_json(out/'passport.json',meta)
            with zipfile.ZipFile(out/'sources.zip','w') as archive:archive.writestr('old/module.py',original)
            checkpoint.verify_frozen_passport(out)
            with zipfile.ZipFile(out/'sources.zip','w') as archive:archive.writestr('old/module.py',b'corrupted')
            checkpoint.verify_execution_passport(out,{}, {})
            with self.assertRaisesRegex(ValueError,'archive corrupted'):checkpoint.verify_frozen_passport(out)
