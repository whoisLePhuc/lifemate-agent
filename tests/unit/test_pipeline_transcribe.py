"""ASR stage helpers that need neither a GPU nor faster-whisper."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lifemate.pipeline import transcribe as t


class AudioPathsTest(unittest.TestCase):
    def test_request_first_then_user_turns_in_order(self):
        item = SimpleNamespace(id='s1', audio='a/s1.wav', history=[
            {'role': 'user', 'audio': 'a/s1_u1.wav'}, {'role': 'assistant', 'text': 'x'},
            {'role': 'user', 'audio': 'a/s1_u2.wav'}])
        self.assertEqual([p.name for p in t.audio_paths(item)], ['s1.wav', 's1_u1.wav', 's1_u2.wav'])

    def test_assemble_keeps_order_and_rejects_empty(self):
        self.assertEqual(t.assemble('s1', ['r', 'h1', 'h2']), {'id': 's1', 'request': 'r', 'history': ['h1', 'h2']})
        with self.assertRaises(ValueError):
            t.assemble('s1', [])


class CudaLibrariesTest(unittest.TestCase):
    def run_ensure(self, current):
        env = {'LD_LIBRARY_PATH': current} if current is not None else {}
        with mock.patch.dict(os.environ, env, clear=False), \
                mock.patch.object(t, 'nvidia_library_path', return_value=['/v/cublas', '/v/cudnn']), \
                mock.patch.object(sys, 'argv', ['lifemate transcribe', '--split', 'public_test']), \
                mock.patch.object(os, 'execve') as execve:
            if current is None:
                os.environ.pop('LD_LIBRARY_PATH', None)
            t.ensure_cuda_libraries()
        return execve

    def test_reexecs_as_a_module_with_the_same_arguments_when_libraries_are_missing(self):
        execve = self.run_ensure('/other')
        argv, env = execve.call_args.args[1], execve.call_args.args[2]
        self.assertEqual(argv, [sys.executable, '-m', 'lifemate.pipeline.transcribe', '--split', 'public_test'])
        self.assertEqual(env['LD_LIBRARY_PATH'], '/v/cublas:/v/cudnn:/other')

    def test_unset_loader_path_gets_only_the_library_directories(self):
        execve = self.run_ensure(None)
        self.assertEqual(execve.call_args.args[2]['LD_LIBRARY_PATH'], '/v/cublas:/v/cudnn')

    def test_no_reexec_when_libraries_are_already_on_the_path(self):
        self.assertFalse(self.run_ensure('/v/cublas:/v/cudnn').called)


class ProfilesTest(unittest.TestCase):
    def test_guarded_profile_has_temperature_fallback_and_greedy_does_not(self):
        self.assertEqual(t.PROFILES['greedy'], {'temperature': 0.0})
        self.assertEqual(t.PROFILES['guarded']['temperature'][0], 0.0)
        self.assertGreater(len(t.PROFILES['guarded']['temperature']), 1)


if __name__ == '__main__':
    unittest.main()
