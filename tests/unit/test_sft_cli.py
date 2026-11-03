import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.train_sft import main


class SFTCLITests(unittest.TestCase):
    def args(self,out):return ['--dataset-dir','unused','--model-path','unused','--output-dir',str(out)]
    def test_existing_partial_output_is_never_modified_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);(out/'provenance.json').write_text('{}',encoding='utf-8')
            with patch('training.train_sft.run',side_effect=ValueError('existing output')):
                with self.assertRaises(ValueError):main(self.args(out))
            self.assertFalse((out/'failure.json').exists())
    def test_failed_new_run_gets_failure_not_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/'new'
            def fail(_):
                out.mkdir();(out/'provenance.json').write_text('{}',encoding='utf-8')
                raise RuntimeError('fixture failure')
            with patch('training.train_sft.run',side_effect=fail):
                with self.assertRaises(RuntimeError):main(self.args(out))
            self.assertEqual(json.loads((out/'failure.json').read_text(encoding='utf-8'))['status'],'failed')
            self.assertFalse((out/'report.json').exists())


if __name__=='__main__':unittest.main()
