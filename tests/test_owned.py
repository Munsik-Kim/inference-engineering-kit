import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from inference_kit.adapters import owned


def handshake(path,process,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if path.exists():return
        if process.poll() is not None:raise AssertionError('Child exited before handshake')
        time.sleep(.01)
    raise AssertionError('Ready handshake deadline exceeded')


@unittest.skipUnless(os.name=='posix' and Path('/proc').exists(),'POSIX /proc owned process adapter')
class OwnedTests(unittest.TestCase):
    def test_graceful_http_ready_and_only_owned_cleanup(self):
        code='''import json,sys
from http.server import HTTPServer,BaseHTTPRequestHandler
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  body=json.dumps({'data':[{'id':'serving-lab'}]} if self.path=='/v1/models' else {'status':'ok'}).encode()
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
 def log_message(self,*args):pass
server=HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler)
server.serve_forever()
'''
        with tempfile.TemporaryDirectory() as t:
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            url=f'http://127.0.0.1:{port}';process,log=owned.start({'argv':[sys.executable,'-B','-u','-c',code,str(port)],'environment':{}},t)
            try:
                observations=[];asyncio.run(owned.ready(process,url,5,observations));self.assertTrue(observations)
                with self.assertRaises(ValueError):asyncio.run(owned.ensure_port_free(url))
            finally:r=owned.stop(process,log,.5)
            self.assertEqual(r['owned_process_cleanup'],'CLEANED');self.assertEqual(r['worker_native_exit']['status'],'UNKNOWN')
            asyncio.run(owned.ensure_port_free(url))
    def test_unresponsive_child_is_killed_after_independent_handshake(self):
        with tempfile.TemporaryDirectory() as t:
            marker=Path(t)/'READY';code='import signal,time,pathlib,sys;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(sys.argv[1]).touch();time.sleep(60)'
            process,log=owned.start({'argv':[sys.executable,'-B','-c',code,str(marker)],'environment':{}},t)
            try:handshake(marker,process)
            finally:
                start=time.monotonic();r=owned.stop(process,log,.2)
            self.assertIn('SIGKILL',r['signals_sent']);self.assertLess(time.monotonic()-start,3)
            self.assertFalse(r['remaining_owned_pids'])
    def test_parent_exit_does_not_skip_grace_for_cleaning_child(self):
        code='''import os,signal,sys,time,pathlib
marker=pathlib.Path(sys.argv[1])
if os.fork()==0:
 signal.signal(signal.SIGINT,lambda *a:(time.sleep(.15),sys.exit(0)))
 marker.touch()
 time.sleep(60)
else:
 signal.signal(signal.SIGINT,lambda *a:sys.exit(0))
 time.sleep(60)
'''
        with tempfile.TemporaryDirectory() as t:
            marker=Path(t)/'READY';process,log=owned.start({'argv':[sys.executable,'-B','-c',code,str(marker)],'environment':{}},t)
            try:handshake(marker,process)
            finally:r=owned.stop(process,log,.7)
            self.assertEqual(r['signals_sent'],['SIGINT']);self.assertFalse(r['remaining_owned_pids'])
    def test_refuses_unowned_process_and_self(self):
        class Unknown:
            pid=os.getpid()
        # WeakKeyDictionary ownership check occurs before touching or signalling PID.
        with self.assertRaisesRegex(ValueError,'unowned'):owned.stop(Unknown(),None,.1)
    def test_readiness_timeout_does_not_retry_launch(self):
        with tempfile.TemporaryDirectory() as t:
            process,log=owned.start({'argv':[sys.executable,'-B','-c','import time;time.sleep(60)'],'environment':{}},t)
            try:
                with self.assertRaises(TimeoutError):asyncio.run(owned.ready(process,'http://127.0.0.1:1',.1))
            finally:self.assertEqual(owned.stop(process,log,.3)['owned_process_cleanup'],'CLEANED')
