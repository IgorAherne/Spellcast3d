# From AURA-V5 src/world_model/utils/profiler_python_utils.py (commit 41007db), adapted for Spellcast3D:
#   - a module logger (log = logging.getLogger(__name__)) instead of the root logging.* calls, so request_profiler.py
#     can route its lines into the tool's log with a "[profiling] " prefix;
#   - _recursive_walk: Spellcast3D's project markers ("spellcast/", "launchers/", "profiling/") instead of AURA's,
#     tested on the path relative to the Spellcast3D folder (outside it nothing is project code: a folder named
#     spellcast above the install must not make the stdlib "ours"); user code is classified on the FULL path
#     (pyinstrument's file_path_short is relative to sys.path, e.g. "threading.py", which made the filter keep the
#     stdlib; a "[self]" node has no path of its own and takes its parent's), the stdlib's own folder counts as a
#     library, while the summary still shows the short path; the registry key uses the full path;
#   - _clean_kernel_name: a branch for stable-diffusion.cpp's (ggml's) CUDA kernels, whose names arrive MANGLED on
#     Windows: readable_kernel_name() demangles free-function and namespaced Itanium names with their template
#     arguments (mul_mat_q<Q4_K,128,false>, quantize_mmq_q8_1<mmq_q8_1_ds_layout=1,false>,
#     k_bin_bcast<op_add,float,float,float>); AURA's heuristics stay for everything else (PyTorch's, cuBLAS's).
#     AURA's buckets would have called mul_mat_q "[Mul Kernel]" and most ggml kernels "[CUDA Kernel]";
#   - process_torch_trace_json: under AURA's GPU BUSY line (kept as it is), one line with the pageable copies' share
#     of it - CUPTI stretches such a copy over the CPU's staging of it, so a request whose weights came from disk read
#     "97% busy" while its first step waited 73 s for host memory.
#   The "AURA" titles inside the summaries are the original's.
import json
import os
import re
import logging
import sysconfig
from collections import defaultdict, Counter

log = logging.getLogger(__name__)

# [SPELLCAST3D] Project code is recognised by these folders inside the Spellcast3D folder (code/spellcast/...).
PROJECT_MARKERS = ("spellcast/", "launchers/", "profiling/")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_REPO_PREFIX = _REPO_ROOT.replace('\\', '/').rstrip('/').lower() + '/'
_STDLIB_PREFIX = (sysconfig.get_paths().get('stdlib') or '').replace('\\', '/').rstrip('/').lower() + '/'


def _repo_relative(norm_path: str):
    """The path below the Spellcast3D folder (forward slashes), or None outside it."""
    lower = norm_path.lower()
    return norm_path[len(_REPO_PREFIX):] if lower.startswith(_REPO_PREFIX) else None


def _recursive_walk(node, registry, parent_full=''):
    func_name = node.get('function', '<unknown>')
    # Robust path lookup: try short, then full, then empty
    file_path = node.get('file_path_short') or node.get('file_path') or ''
    # [SPELLCAST3D] Classified on the full path; a "[self]" node has none and is its parent's own time.
    full_path = node.get('file_path') or (parent_full if func_name == '[self]' else '') or file_path
    
    # Filter Logic: Identify User Code
    norm_path = full_path.replace('\\', '/')
    
    # [ADJUSTMENT] Broader library detection to filter out Ray internals
    is_library = any(x in norm_path for x in ["site-packages", "dist-packages", "lib/python", "ray/"])
    # [SPELLCAST3D] The stdlib wherever this Python keeps it (a portable or uv-managed Python's Lib folder).
    is_library = is_library or (_STDLIB_PREFIX != '/' and norm_path.lower().startswith(_STDLIB_PREFIX))
    
    # [ADJUSTMENT] Explicitly whitelist your project folders
    # [SPELLCAST3D] Spellcast3D's folders, below the Spellcast3D folder only (its venvs are libraries).
    relative = _repo_relative(norm_path)
    is_project = relative is not None and not is_library and any(x in relative for x in PROJECT_MARKERS)
    
    # Final logic: It is user code if it is in the project OR (it is not a library AND not python internals)
    is_user_code = is_project or (not is_library and "python" not in norm_path.lower())

    key = f"{func_name}::{full_path}"
    
    time_inclusive = node.get('time', 0.0)
    children_time = sum(c.get('time', 0.0) for c in node.get('children', []))
    time_exclusive = max(0.0, time_inclusive - children_time)
    
    if key not in registry:
        registry[key] = {
            'function': func_name,
            'file': file_path,
            'total_time': 0.0,
            'self_time': 0.0,
            'calls': 0,
            'is_user_code': is_user_code
        }
    
    registry[key]['total_time'] += time_inclusive
    registry[key]['self_time'] += time_exclusive
    registry[key]['calls'] += 1

    for child in node.get('children', []):
        _recursive_walk(child, registry, full_path)

def process_profile_json(json_path: str, output_txt_path: str):
    """
    Reads a PyInstrument JSON file, aggregates stats, and writes a summary.
    """
    if not os.path.exists(json_path):
        log.error(f"Profiler JSON not found: {json_path}")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content:
                log.error("Profiler JSON file is empty.")
                return
            data = json.loads(content)
            
        root_frame = data.get('root_frame', data)
        registry = {}
        _recursive_walk(root_frame, registry)
        
        # Filter for User Code
        stats_list = [s for s in registry.values() if s['is_user_code']]
        stats_list.sort(key=lambda x: x['total_time'], reverse=True)
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"--- AURA PERFORMANCE SUMMARY ---\n")
            f.write(f"Source: {json_path}\n")
            f.write(f"FUNCTION | TOTAL (s) | SELF (s) | CALLS | LOCATION\n")
            f.write("---\n")
            
            for item in stats_list[:50]:
                func = item['function']
                total = f"{item['total_time']:.4f}"
                self_t = f"{item['self_time']:.4f}"
                calls = item['calls']
                loc = item['file']
                # Compact format: No fixed width padding
                f.write(f"{func} | {total} | {self_t} | {calls} | {loc}\n")
                
        log.info(f">>> AURA PROFILER: Summary generated at {output_txt_path}")
        
    except Exception as e:
        log.error(f">>> AURA PROFILER: Error generating summary: {e}")


# ---- [SPELLCAST3D] readable names for stable-diffusion.cpp's (ggml's) CUDA kernels ----------------------------------
# CUPTI reports kernel names mangled on Windows (Linux's Kineto demangles them), e.g.
#   _Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_PKf5uint3iiiiiS8_S8_iiiS8_S8_iiiS8_
# Just enough of the Itanium C++ ABI is read here for kernels: the (namespaced) function name and its template
# arguments - types, integer / bool / enum literals, &function. The parameters after them are not shown (they say
# nothing the template arguments do not). Anything else is _Unreadable and keeps AURA's heuristic bucket.

# ggml.h's enum ggml_type (checked against ggml-org/ggml master, Oct 2026; 4, 5, 31-33 and 36-38 were removed).
GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
              26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
              40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}
# Kernels of these libraries keep AURA's buckets ([GEMM Kernel], [Elementwise: FillFunctor], ...): their full
# template arguments run to hundreds of characters.
LIBRARY_NAMESPACES = ("at", "c10", "torch", "cutlass", "cute", "cub", "thrust", "cudnn", "cublas", "cublasLt",
                      "flash", "fmha", "xformers", "triton", "std")
_BUILTIN_TYPES = {"v": "void", "w": "wchar_t", "b": "bool", "c": "char", "a": "signed char", "h": "unsigned char",
                  "s": "short", "t": "unsigned short", "i": "int", "j": "unsigned int", "l": "long",
                  "m": "unsigned long", "x": "long long", "y": "unsigned long long", "n": "__int128",
                  "o": "unsigned __int128", "f": "float", "d": "double", "e": "long double", "g": "__float128",
                  "z": "..."}
_BUILTIN_D = {"Dh": "half", "Di": "char32_t", "Ds": "char16_t", "Du": "char8_t", "Da": "auto",
              "Dc": "decltype(auto)", "Dn": "std::nullptr_t"}
_STD_ABBREVIATIONS = {"Sa": "std::allocator", "Sb": "std::basic_string", "Ss": "std::string",
                      "Si": "std::istream", "So": "std::ostream", "Sd": "std::iostream"}
_FLOAT_N = re.compile(r"DF(\d+)([_bx])")
# nvcc puts a .cu file's static functions into a namespace of its own, e.g. _INTERNAL_d3f459f4_11_binbcast_cu_6840010b
# (op_add of k_bin_bcast<op_add,...>): left out of the readable name.
_NVCC_INTERNAL = re.compile(r"_INTERNAL_[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+")
_NUMBER = re.compile(r"n?\d+")


class _Unreadable(Exception):
    """A mangled name outside the subset _Demangler reads."""


class _Demangler:
    """Itanium-ABI names of CUDA kernels -> (qualified name parts, template arguments or None). Tracks the
    substitution table (S_, S0_, ...) the way the ABI numbers it, since template arguments refer back to it."""

    def __init__(self, text: str):
        self.s, self.i = text, 0
        self.subs = []            # substitution candidates, in the ABI's order
        self.tparams = []         # the template arguments T_, T0_, ... refer to (the innermost function's)

    def kernel(self):
        """The kernel's name; its return and parameter types are read too (and dropped): a name that does not
        parse to its very end was misread, and keeps AURA's bucket instead of showing something wrong."""
        self._expect("_Z")
        parts, args = self._name()
        if args is not None:
            self.tparams = args
        while self.i < len(self.s):
            self._type()
        return parts, args

    # -- tokens
    def _peek(self, n=1):
        return self.s[self.i:self.i + n]

    def _eat(self, token):
        if self.s.startswith(token, self.i):
            self.i += len(token)
            return True
        return False

    def _expect(self, token):
        if not self._eat(token):
            raise _Unreadable(f"expected {token!r} at {self.i}")

    def _source_name(self):
        m = re.compile(r"\d+").match(self.s, self.i)
        if not m:
            raise _Unreadable(f"expected a length at {self.i}")
        n = int(m.group())
        name = self.s[m.end():m.end() + n]
        if len(name) != n:
            raise _Unreadable("truncated name")
        self.i = m.end() + n
        if _NVCC_INTERNAL.fullmatch(name):
            return ""                                    # nvcc's namespace for a .cu file's statics: not shown
        return "(anonymous namespace)" if name.startswith("_GLOBAL__N") else name

    @staticmethod
    def _args_text(args):
        return "<" + ",".join(args) + ">"

    @staticmethod
    def _join(parts):
        return "::".join(part for part in parts if part)

    # -- names
    def _unqualified(self):
        self._eat("L")                                   # internal linkage (a static function)
        if not self._peek().isdigit():
            raise _Unreadable(f"unsupported name at {self.i}")
        name = self._source_name()
        while self._eat("B"):                            # ABI tags
            self._source_name()
        return name

    def _name(self, is_type=False):
        """(parts, final template arguments or None). A function's own name is no substitution candidate, its
        template's name is; for a type both are."""
        c = self._peek()
        if c == "N":
            return self._nested(is_type)
        if self._eat("St"):
            parts = ["std", self._unqualified()]
        elif c == "S":
            parts = [self._substitution()]
            if self._peek() != "I":
                return parts, None
            args = self._template_args()
            if is_type:
                self.subs.append(parts[0] + self._args_text(args))
            return parts, args
        else:
            parts = [self._unqualified()]
        qualified = self._join(parts)
        if self._peek() == "I":
            self.subs.append(qualified)
            args = self._template_args()
            if is_type:
                self.subs.append(qualified + self._args_text(args))
            return parts, args
        if is_type:
            self.subs.append(qualified)
        return parts, None

    def _nested(self, is_type):
        self._expect("N")
        while self._peek() in ("r", "V", "K", "R", "O"):     # a member function's qualifiers
            self.i += 1
        parts, base, final_args = [], "", None
        while not self._eat("E"):
            c = self._peek()
            if c == "I":
                if not parts:
                    raise _Unreadable("template arguments without a name")
                args = self._template_args()
                parts[-1] = base + self._args_text(args)
                final_args = args
                if self._peek() != "E" or is_type:
                    self.subs.append(self._join(parts))
                continue
            final_args = None
            if c == "S" and not parts:
                if self._eat("St"):
                    parts = ["std"]
                else:
                    parts = [self._substitution()]
                base = parts[-1]
                continue
            if c == "T" and not parts:
                parts = [self._template_param()]
                base = parts[-1]
                self.subs.append(parts[0])
                continue
            if c in ("C", "D", "U") or not c:
                raise _Unreadable(f"constructor, destructor or unnamed entity at {self.i}")
            base = self._unqualified()
            parts.append(base)
            if self._peek() != "E" or is_type:
                self.subs.append(self._join(parts))
        if not parts:
            raise _Unreadable("empty nested name")
        if final_args is not None:
            parts[-1] = base
        return parts, final_args

    def _substitution(self):
        self._expect("S")
        end = self.s.find("_", self.i)
        if end < 0:
            raise _Unreadable("unterminated substitution")
        seq = self.s[self.i:end]
        self.i = end + 1
        try:
            index = int(seq, 36) + 1 if seq else 0
        except ValueError:
            raise _Unreadable(f"bad substitution {seq!r}")
        if index >= len(self.subs):
            raise _Unreadable(f"substitution {index} of {len(self.subs)}")
        return self.subs[index]

    def _template_param(self):
        self._expect("T")
        end = self.s.find("_", self.i)
        if end < 0:
            raise _Unreadable("unterminated template parameter")
        seq = self.s[self.i:end]
        self.i = end + 1
        index = int(seq) + 1 if seq.isdigit() else 0
        return self.tparams[index] if index < len(self.tparams) else f"T{index}"

    # -- template arguments
    def _template_args(self):
        self._expect("I")
        args = []
        while not self._eat("E"):
            arg = self._template_arg()
            if arg is not None:
                args.append(arg)
        return args

    def _template_arg(self):
        c = self._peek()
        if c == "L":
            return self._literal()
        if c == "X":
            self.i += 1
            value = self._expression()
            self._expect("E")
            return value
        if c == "J":                                      # a parameter pack; empty ones vanish
            self.i += 1
            items = []
            while not self._eat("E"):
                item = self._template_arg()
                if item is not None:
                    items.append(item)
            return ",".join(items) if items else None
        return self._type()

    def _literal(self):
        self._expect("L")
        if self._eat("_Z"):                               # an external name: a function's address
            name = self._encoding()
            self._expect("E")
            return name
        if self._eat("Dn"):
            self._literal_value()
            self._expect("E")
            return "nullptr"
        c = self._peek()
        if c in _BUILTIN_TYPES:
            self.i += 1
            value = self._literal_value()
            self._expect("E")
            if c == "b":
                return "true" if value == "1" else "false"
            return "-" + value[1:] if value.startswith("n") else value
        enum = self._type()                               # an enum's value, e.g. L9ggml_type12E
        value = self._literal_value()
        self._expect("E")
        if not _NUMBER.fullmatch(value):
            raise _Unreadable(f"literal {value!r}")
        number = int(value.replace("n", "-"))
        if enum == "ggml_type" and number in GGML_TYPES:
            return GGML_TYPES[number]
        return f"{enum}={number}"

    def _literal_value(self):
        end = self.s.find("E", self.i)
        if end < 0:
            raise _Unreadable("unterminated literal")
        value = self.s[self.i:end]
        self.i = end
        return value

    def _expression(self):
        if self._eat("ad"):                               # &function: shown as the function's name
            return self._expression()
        c = self._peek()
        if c == "L":
            return self._literal()
        if c == "T":
            return self._template_param()
        raise _Unreadable(f"unsupported expression at {self.i}")

    def _encoding(self):
        """A function inside a template argument (L_Z...E): its name, then its parameter types (skipped, but they
        are substitution candidates)."""
        parts, args = self._name()
        outer = self.tparams
        if args is not None:
            self.tparams = args
        while self._peek() not in ("E", ""):
            self._type()
        self.tparams = outer
        return self._join(parts) + (self._args_text(args) if args is not None else "")

    # -- types
    def _type(self):
        s, i = self.s, self.i
        c = self._peek()
        if not c:
            raise _Unreadable("name ends inside a type")
        if c in _BUILTIN_TYPES:
            self.i += 1
            return _BUILTIN_TYPES[c]
        if c == "u":                                      # a vendor's type
            self.i += 1
            name = self._source_name()
            self.subs.append(name)
            return name
        if c == "D":
            two = s[i:i + 2]
            if two in _BUILTIN_D:
                self.i += 2
                return _BUILTIN_D[two]
            m = _FLOAT_N.match(s, i)
            if m:
                self.i = m.end()
                return "__bf16" if m.group(2) == "b" else f"_Float{m.group(1)}"
            if two == "Dp":                               # a pack expansion
                self.i += 2
                text = self._type() + "..."
                self.subs.append(text)
                return text
            if two == "Dv":
                m = re.compile(r"Dv(\d+)_").match(s, i)
                if not m:
                    raise _Unreadable("vector type")
                self.i = m.end()
                text = f"{self._type()} vector[{m.group(1)}]"
                self.subs.append(text)
                return text
            raise _Unreadable(f"unsupported type at {i}")
        if c in ("r", "V", "K"):
            quals = []
            while self._peek() in ("r", "V", "K"):
                quals.append({"r": "restrict", "V": "volatile", "K": "const"}[self._peek()])
                self.i += 1
            text = " ".join(reversed(quals)) + " " + self._type()
            self.subs.append(text)
            return text
        if c in ("P", "R", "O"):
            self.i += 1
            text = self._type() + {"P": "*", "R": "&", "O": "&&"}[c]
            self.subs.append(text)
            return text
        if c == "F":
            self.i += 1
            self._eat("Y")
            ret = self._type()
            params = []
            while not self._eat("E"):
                if self._peek() in ("R", "O") and self._peek(2)[1:] == "E":
                    self.i += 1
                    continue
                params.append(self._type())
            text = f"{ret}({','.join(params)})"
            self.subs.append(text)
            return text
        if c == "A":
            m = re.compile(r"A(\d*)_").match(s, i)
            if not m:
                raise _Unreadable("array type")
            self.i = m.end()
            text = f"{self._type()}[{m.group(1)}]"
            self.subs.append(text)
            return text
        if c == "M":
            self.i += 1
            owner = self._type()
            member = self._type()
            text = f"{member} {owner}::*"
            self.subs.append(text)
            return text
        if c == "T":
            text = self._template_param()
            self.subs.append(text)
            if self._peek() == "I":
                text += self._args_text(self._template_args())
                self.subs.append(text)
            return text
        if c == "S":
            if self._eat("St"):
                name = "std::" + self._unqualified()
                self.subs.append(name)
                if self._peek() == "I":
                    name += self._args_text(self._template_args())
                    self.subs.append(name)
                return name
            abbreviation = _STD_ABBREVIATIONS.get(s[i:i + 2])
            if abbreviation:
                self.i += 2
                text = abbreviation
            else:
                text = self._substitution()
            if self._peek() == "I":
                text += self._args_text(self._template_args())
                self.subs.append(text)
            return text
        if c == "N" or c.isdigit():
            parts, args = self._name(is_type=True)
            return self._join(parts) + (self._args_text(args) if args is not None else "")
        raise _Unreadable(f"unsupported type at {i}")


def readable_kernel_name(name: str):
    """A mangled kernel name as its function's name with the template arguments, e.g.
    mul_mat_q<Q4_K,128,false> - or None when it is not a mangled name this reads, or when it belongs to a library
    in LIBRARY_NAMESPACES (PyTorch's, CUTLASS's: AURA's heuristics name those)."""
    if not isinstance(name, str) or not name.startswith("_Z"):
        return None
    try:
        parts, args = _Demangler(name).kernel()
    except (_Unreadable, ValueError, IndexError, RecursionError):
        return None
    shown = [part for part in parts if part]
    if not shown or (len(shown) > 1 and shown[0] in LIBRARY_NAMESPACES):
        return None
    return "::".join(shown) + (_Demangler._args_text(args) if args is not None else "")


def _clean_kernel_name(name: str) -> str:
    """
    Simplifies C++ mangled names (starting with _Z) into readable categories 
    to save tokens, while keeping Python paths and meaningful names intact.
    """
    if not name or not isinstance(name, str):
        return "unknown"
        
    # Pass through Python files, aten:: ops, and standard names
    if not name.startswith("_Z"):
        # Optional: Clean up standard wrapper noise if needed
        if "pybind11" in name: return "[pybind11_wrapper]"
        return name

    # [SPELLCAST3D] stable-diffusion.cpp's ggml kernels (free functions, or namespaced outside the libraries): the
    # function with its template arguments, which tell the quantization types and tile sizes apart. Distinct
    # instantiations stay distinct rows (the callers key their rows by the raw name).
    readable = readable_kernel_name(name)
    if readable is not None:
        return readable

    # Heuristic mapping for mangled CUDA kernels
    lower_name = name.lower()
    if "gemm" in lower_name: return "[GEMM Kernel]"
    if "convolution" in lower_name or "conv" in lower_name: return "[Conv Kernel]"
    if "elementwise" in lower_name:
        # Keep the op: the mangled name carries the functor (MulFunctor, sin_kernel_cuda, ...),
        # and a single "[Elementwise Kernel]" bucket hid five different ops at 8.8s/15s of GPU.
        m = re.search(r'([A-Za-z]+Functor|[a-z_]+_kernel_cuda|[a-z_]+_kernel_impl)', name)
        return f"[Elementwise: {m.group(1)}]" if m else "[Elementwise Kernel]"
    if "layer_norm" in lower_name: return "[LayerNorm Kernel]"
    if "softmax" in lower_name: return "[Softmax Kernel]"
    if "copy" in lower_name: return "[Copy Kernel]"
    if "add" in lower_name: return "[Add Kernel]"
    if "mul" in lower_name: return "[Mul Kernel]"
    if "activation" in lower_name or "silu" in lower_name or "relu" in lower_name: return "[Activation Kernel]"
    if "fft" in lower_name: return "[FFT Kernel]"
    if "scatter" in lower_name: return "[Scatter Kernel]"
    if "gather" in lower_name: return "[Gather Kernel]"
    if "index" in lower_name: return "[Indexing Kernel]"
    if "vectorized" in lower_name: return "[Vectorized Kernel]"
    
    return "[CUDA Kernel]"


def process_torch_trace_json(json_path: str, output_txt_path: str, top_k=50):
    """
    Reads a Torch Profiler Chrome Trace JSON.
    [FIXED] Sanitizes bad JSON, calculates Self Time, and aggregates SHAPES.
    """
    if not os.path.exists(json_path):
        return

    try:
        # 1. Read and Sanitize (Fix the """ bug)
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        if '"""' in raw_content:
            log.warning(f"Sanitizing corrupt Torch trace: {json_path}")
            raw_content = raw_content.replace('"""', "'''") 
            # Optional: Save back to disk so Chrome can open it
            # with open(json_path, 'w', encoding='utf-8') as f: f.write(raw_content)

        data = json.loads(raw_content)
    except Exception as e:
        log.error(f"Error reading Torch trace {json_path}: {e}")
        return

    events = data.get('traceEvents', [])
    # Filter for duration events ('X') 
    events = [e for e in events if e.get('ph') == 'X' and 'dur' in e]
    
    if not events:
        return

    # Ties broken longest-first: a parent opening on the same microsecond as its child must be
    # pushed first, or the nesting inverts.
    events.sort(key=lambda x: (x['ts'], -x['dur']))

    # Stats now tracks 'shapes' as a Counter
    stats = defaultdict(lambda: {
        'total': 0.0, 
        'self': 0.0, 
        'calls': 0, 
        'callers': Counter(),
        'shapes': Counter() 
    })

    # ONE STACK PER (pid, tid). A single global stack over timestamp-sorted events nests every thread
    # inside whatever happened to be open on another one: with the loader's prefetch threads live, a
    # blocking recv "contained" the training step it merely overlapped. TOTAL and CALLS were always
    # right; SELF and the caller lines are only right after this.
    stacks = defaultdict(list)
    gpu_busy = 0.0                      # DEVICE activity only -- see the GPU BUSY note below
    pageable_us = 0.0                   # [SPELLCAST3D] the pageable copies inside gpu_busy
    dispatch_count = Counter()          # (pid,tid) -> cpu_op + cuda_runtime event COUNT
    thread_cat = defaultdict(Counter)   # (pid,tid) -> categories, so the table can name the thread
    steps = {}                          # ProfilerStep#N -> [min_start, max_end]

    for e in events:
        start = e['ts']
        dur = e['dur']
        end = start + dur
        name = e.get('name', 'unknown')

        # --- NEW: Extract Shapes ---
        # Torch usually stores shapes in args['Input Dims'] as a list of lists/ints
        # We stringify it to make it hashable for the Counter.
        args = e.get('args', {})
        shape_str = ""
        if 'Input Dims' in args:
            # Format: [[32, 64], [64, 128]]
            shape_info = args['Input Dims']
            # Make it compact: remove spaces
            shape_str = str(shape_info).replace(" ", "")

        tkey = (e.get('pid'), e.get('tid'))
        cat = e.get('cat', '?')
        # record_function spans are emitted once on the CPU track and again on every GPU track
        # they touch ('gpu_user_annotation'). The GPU copy is what kernels nest under, so it must
        # stay on the stack for caller attribution -- but it is not a second call and not more
        # time: counting it made C1_Warmup_Forward read 88s TOTAL inside a 19s window.
        is_gpu_annotation = cat == 'gpu_user_annotation'
        if not is_gpu_annotation:
            stats[name]['total'] += dur
            stats[name]['calls'] += 1
            stats[name]['self'] += dur
            if shape_str:
                stats[name]['shapes'][shape_str] += 1

        thread_cat[tkey][cat] += 1
        # DEVICE activity ONLY. The step's gpu_user_annotation spans the whole step on the GPU
        # timeline, so a "top-level event time per thread" table reads ~95% busy on an idle GPU.
        # Over-counts if kernels ever run on concurrent streams; single-stream here.
        if cat in ('kernel', 'gpu_memcpy', 'gpu_memset'):
            gpu_busy += dur
            if cat == 'gpu_memcpy' and 'pageable' in name.lower():
                pageable_us += dur      # [SPELLCAST3D] see the note under GPU BUSY
        elif cat in ('cpu_op', 'cuda_runtime'):
            dispatch_count[tkey] += 1
        if name.startswith('ProfilerStep#'):
            s = steps.setdefault(name, [start, end])
            s[0] = min(s[0], start); s[1] = max(s[1], end)

        # Stack Logic for Self Time
        stack = stacks[tkey]
        while stack and stack[-1]['end'] <= start:
            stack.pop()

        # Annotations are mirrors, not calls: they never drain a counted parent (a full-span
        # copy under ProfilerStep# would zero its self) and never record a row of their own.
        if not is_gpu_annotation and stack:
            parent = stack[-1]
            # Children subtract from a parent's self only when that parent's time was counted:
            # under a GPU annotation, subtracting kernel time would drain the CPU copy's self.
            if parent['count_self']:
                stats[parent['name']]['self'] -= dur
                if stats[parent['name']]['self'] < 0: stats[parent['name']]['self'] = 0
            stats[name]['callers'][parent['name']] += dur

        stack.append({'name': name, 'end': end, 'count_self': not is_gpu_annotation})

    # Prepare Results
    results = []
    for name, data in stats.items():
        # Get most common shape
        most_common_shape = ""
        if data['shapes']:
            # Grab the shape that appears most frequently
            most_common_shape, _ = data['shapes'].most_common(1)[0]
            # Truncate if insanely long (rare, but good for safety)
            if len(most_common_shape) > 40:
                most_common_shape = most_common_shape[:37] + "..."

        results.append({
            'name': _clean_kernel_name(name),
            'self_ms': data['self'] / 1000.0,
            'total_ms': data['total'] / 1000.0,
            'calls': data['calls'],
            'top_callers': data['callers'].most_common(3),
            'shape': most_common_shape
        })

    results.sort(key=lambda x: x['self_ms'], reverse=True)

    # Window = the ProfilerStep span when the caller stepped the schedule, else the whole trace.
    spans = sorted(steps.values()) if steps else None
    if spans:
        window_us = max(max(b for _, b in spans) - min(a for a, _ in spans), 1.0)
    else:
        window_us = max(max(e['ts'] + e['dur'] for e in events)
                        - min(e['ts'] for e in events), 1.0)

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"--- AURA TORCH KERNEL SUMMARY ---\n")
        f.write(f"Source: {json_path}\n")
        # READ THIS FIRST. The active window is a handful of steps. A bursty loader (the shuffle pool
        # drains capacity/batch ready batches back-to-back) makes that window run far faster than the
        # run's own it/s: the trace measures STEP COST honestly and says NOTHING about wall-clock,
        # which is [pipe]'s question.
        if spans:
            f.write(f"PROFILED WINDOW: {len(spans)} step(s) over {window_us / 1000.0:.1f} ms = "
                    f"{1e6 * len(spans) / window_us:.1f} it/s IN-WINDOW "
                    f"(compare against the run's own it/s)\n")
            f.write("  per-step ms: " + " ".join(f"{(b - a) / 1000.0:.1f}" for a, b in spans) + "\n")
        # The one number separating "model too slow" from "starved". Single digits = the trainer
        # is waiting for data and no step-side change can matter. NOT inflated by with_stack:
        # device timings come from CUPTI, not the python tracer.
        f.write(f"GPU BUSY: {gpu_busy / 1000.0:.1f} ms of {window_us / 1000.0:.1f} ms "
                f"({100.0 * gpu_busy / window_us:.1f}%)  [kernel+memcpy+memset]\n")
        if pageable_us > 0:
            # [SPELLCAST3D] CUPTI stretches a pageable copy over the CPU's staging of it (page faults from disk too)
            f.write(f"  [SPELLCAST3D] of it pageable copies: {pageable_us / 1000.0:.1f} ms, mostly the CPU staging "
                    f"them, not GPU work; without them {(gpu_busy - pageable_us) / 1000.0:.1f} ms "
                    f"({100.0 * (gpu_busy - pageable_us) / window_us:.1f}%). cuda_trace_report counts them as "
                    f"waiting for host memory, not busy.\n")
        # Added INPUT SHAPES column
        f.write(f"FUNCTION/KERNEL | SELF (ms) | TOTAL (ms) | CALLS | INPUT SHAPES (Top)\n")
        f.write("---\n")
        
        for r in results[:top_k]:
            # Compact printing: Just the shape string, no labels
            shape_display = r['shape'] if r['shape'] else "-"
            f.write(f"{r['name']} | {r['self_ms']:.2f} | {r['total_ms']:.2f} | {r['calls']} | {shape_display}\n")
            
            if not r['top_callers']:
                continue
            for caller_name, caller_dur_us in r['top_callers']:
                caller_ms = caller_dur_us / 1000.0
                pct = (caller_ms / r['total_ms']) * 100 if r['total_ms'] > 0 else 0
                clean_caller = _clean_kernel_name(caller_name)
                clean_caller = clean_caller.replace("enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", "DataLoader")
                f.write(f" {clean_caller} ({caller_ms:.1f}ms - {pct:.0f}%)\n")

        # WHICH THREAD drives the device. COUNTS, not durations: cpu_op events NEST, so summing
        # durations double-counts while the count still names the dispatching thread. A loader thread
        # reads ~0 here even when it burns the whole window inside recv -- which is the point.
        f.write("---\nTORCH DISPATCH by thread (cpu_op + cuda_runtime event count)\n")
        for (pid, tid), n in dispatch_count.most_common(8):
            cats = ",".join(k for k, _ in thread_cat[(pid, tid)].most_common(2))
            f.write(f" pid {pid} tid {tid} [{cats}]: {n} events\n")

    log.info(f">>> AURA PROFILER: Torch Summary generated at {output_txt_path}")
