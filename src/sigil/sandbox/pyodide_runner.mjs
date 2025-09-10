// Node-based Pyodide runner.
// Reads a single JSON request from stdin and writes a single JSON line to stdout.
// The request shape:
//   { code: str, func_name: str, args: list, kwargs: dict, limits: {...} }
// It requires a local Pyodide distribution. Configure via env:
//   PYODIDE_INDEX_URL=file:///absolute/path/to/pyodide
// If not set, defaults to a "pyodide/" folder next to this script.

import { stdin, stdout } from 'node:process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const MAX_OUT = 64 * 1024;

function readStdinOnce() {
  return new Promise((resolve) => {
    let data = '';
    stdin.setEncoding('utf8');
    stdin.on('data', (chunk) => (data += chunk));
    stdin.on('end', () => resolve(data));
  });
}

function safeWrite(obj) {
  const s = JSON.stringify(obj);
  stdout.write(s.slice(0, MAX_OUT));
  stdout.write('\n');
}

try {
  const raw = await readStdinOnce();
  const req = JSON.parse(raw || '{}');

  // Resolve indexURL
  let indexURL = process.env.PYODIDE_INDEX_URL;
  if (!indexURL) {
    const here = dirname(fileURLToPath(import.meta.url));
    const local = `file://${here}/pyodide`;
    // We can't reliably check file:// with existsSync on all OS; try path form
    const pathGuess = `${here}/pyodide`;
    if (existsSync(pathGuess)) {
      indexURL = local;
    }
  }
  // If indexURL is not provided, loadPyodide() will use CDN. This requires
  // network access. Set PYODIDE_INDEX_URL to a local file:// path for offline.

  let loadPyodide;
  try {
    ({ loadPyodide } = await import('pyodide')); // npm package must be installed
  } catch (e) {
    safeWrite({ ok: false, error: 'Cannot import pyodide npm package. Install it or vendor assets.', detail: String(e) });
    process.exit(0);
  }

  let pyodide;
  try {
    pyodide = indexURL ? await loadPyodide({ indexURL }) : await loadPyodide();
  } catch (e) {
    safeWrite({ ok: false, error: `Failed to load Pyodide at indexURL=${indexURL}`, detail: String(e) });
    process.exit(0);
  }

  // Build the invocation helper in Python
  const helperCode = `
import sys, io, types, traceback
def _sigil_invoke(code, func_name, args, kwargs):
    stdout = io.StringIO(); stderr = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = stdout, stderr
        ns = {}
        exec(code, ns)
        fn = ns.get(func_name)
        if not callable(fn):
            raise RuntimeError(f"function '{func_name}' not found after exec")
        res = fn(*args, **kwargs)
        ok = True
        err = None
    except Exception:
        ok = False
        res = None
        err = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return ok, res, stdout.getvalue(), stderr.getvalue(), err
`;

  await pyodide.runPythonAsync(helperCode);
  const invoke = pyodide.globals.get('_sigil_invoke');

  // Convert inputs using pyodide conversions
  const toPy = pyodide.toPy.bind(pyodide);
  const argsPy = toPy(req.args || []);
  const kwargsPy = toPy(req.kwargs || {});

  let resultTuple;
  try {
    resultTuple = invoke(req.code || '', req.func_name || '', argsPy, kwargsPy);
  } catch (e) {
    safeWrite({ ok: false, error: `Invocation failed: ${String(e)}` });
    process.exit(0);
  }

  // Convert back to JS
  const js = resultTuple.toJs({ dict_converter: Object });
  const [ok, result, outStr, errStr, err] = js;
  safeWrite({ ok: !!ok, result, stdout: outStr || '', stderr: errStr || '', error: err || null });
} catch (e) {
  safeWrite({ ok: false, error: String(e) });
}
