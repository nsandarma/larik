#!/usr/bin/env python3
"""
End-to-end small Clang JIT pipeline:
 - compile C source to .o via clang (ClangJITCompiler)
 - parse ELF object (elf_loader)
 - apply relocations (jit_loader + relocate)
 - map final image to executable memory (RW -> RX) and call function via ctypes

Notes / limitations:
 - Minimal subset of ELF structures implemented via ctypes (Elf64).
 - Supports basic R_X86_64_PC32 and several AArch64 relocations shown earlier.
 - Only handles simple progbits sections and .symtab; no dynamic linking, GOT, PLT, TLS, etc.
 - Requires clang installed and accessible as $CC or "clang".
 - Might still segfault for complex code or missing reloc types.
"""

import os
import sys
import platform
import subprocess
import struct
import mmap
import ctypes
from dataclasses import dataclass
from typing import List, Tuple

# -----------------------
# Helpers (bit ops)
# -----------------------
def getbits(x: int, lo: int, hi: int) -> int:
    """Return bits [lo..hi] inclusive (lo..hi-1 if used like earlier).
    The earlier code used getbits(x, start, end) where end is exclusive.
    We'll implement as getbits(x, lo, hi_excl).
    """
    mask = (1 << (hi - lo)) - 1
    return (x >> lo) & mask

def i2u(bits: int, v: int) -> int:
    """Convert signed int value to unsigned of given bit width (two's complement)."""
    mask = (1 << bits) - 1
    return v & mask

# -----------------------
# Minimal ELF64 ctypes structs & constants
# (fields needed by loader only)
# -----------------------
import ctypes.util

# constants
SHT_NULL = 0
SHT_PROGBITS = 1
SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_REL = 9

# relocation types constants (we'll only reference the ones we use)
# R_X86_64_PC32 = 2 (typical)
R_X86_64_PC32 = 2

# AArch64 relocation constants (common names)
R_AARCH64_ADR_PREL_PG_HI21 = 1030
R_AARCH64_ADD_ABS_LO12_NC = 1033
R_AARCH64_LDST16_ABS_LO12_NC = 1040
R_AARCH64_LDST32_ABS_LO12_NC = 1041
R_AARCH64_LDST64_ABS_LO12_NC = 1042
R_AARCH64_LDST128_ABS_LO12_NC = 1043

# ELF structures (very small subset)
class Elf64_Ehdr(ctypes.Structure):
    _fields_ = [
        ("e_ident", ctypes.c_ubyte * 16),
        ("e_type", ctypes.c_uint16),
        ("e_machine", ctypes.c_uint16),
        ("e_version", ctypes.c_uint32),
        ("e_entry", ctypes.c_uint64),
        ("e_phoff", ctypes.c_uint64),
        ("e_shoff", ctypes.c_uint64),
        ("e_flags", ctypes.c_uint32),
        ("e_ehsize", ctypes.c_uint16),
        ("e_phentsize", ctypes.c_uint16),
        ("e_phnum", ctypes.c_uint16),
        ("e_shentsize", ctypes.c_uint16),
        ("e_shnum", ctypes.c_uint16),
        ("e_shstrndx", ctypes.c_uint16),
    ]

class Elf64_Shdr(ctypes.Structure):
    _fields_ = [
        ("sh_name", ctypes.c_uint32),
        ("sh_type", ctypes.c_uint32),
        ("sh_flags", ctypes.c_uint64),
        ("sh_addr", ctypes.c_uint64),
        ("sh_offset", ctypes.c_uint64),
        ("sh_size", ctypes.c_uint64),
        ("sh_link", ctypes.c_uint32),
        ("sh_info", ctypes.c_uint32),
        ("sh_addralign", ctypes.c_uint64),
        ("sh_entsize", ctypes.c_uint64),
    ]

class Elf64_Sym(ctypes.Structure):
    _fields_ = [
        ("st_name", ctypes.c_uint32),
        ("st_info", ctypes.c_ubyte),
        ("st_other", ctypes.c_ubyte),
        ("st_shndx", ctypes.c_uint16),
        ("st_value", ctypes.c_uint64),
        ("st_size", ctypes.c_uint64),
    ]

# Rel and Rela entries
class Elf64_Rel(ctypes.Structure):
    _fields_ = [
        ("r_offset", ctypes.c_uint64),
        ("r_info", ctypes.c_uint64),
    ]

class Elf64_Rela(ctypes.Structure):
    _fields_ = [
        ("r_offset", ctypes.c_uint64),
        ("r_info", ctypes.c_uint64),
        ("r_addend", ctypes.c_int64),
    ]

# Helpers to extract sym/index/type from r_info (ELF64)
def ELF64_R_SYM(r_info: int) -> int:
    return r_info >> 32

def ELF64_R_TYPE(r_info: int) -> int:
    return r_info & 0xFFFFFFFF

# -----------------------
# ElfSection dataclass
# -----------------------
@dataclass
class ElfSection:
    name: str
    header: Elf64_Shdr
    content: bytes

# -----------------------
# elf_loader implementation
# -----------------------
def _strtab(blob: bytes, idx: int) -> str:
    if idx >= len(blob):
        return ""
    end = blob.find(b'\x00', idx)
    if end == -1:
        end = len(blob)
    return blob[idx:end].decode('utf-8')

def elf_loader(blob: bytes, force_section_align:int=1) -> tuple[memoryview, List[ElfSection], List[tuple], List[Elf64_Sym]]:
    # parse ELF header
    if len(blob) < ctypes.sizeof(Elf64_Ehdr):
        raise RuntimeError("blob too small for ELF header")
    header = Elf64_Ehdr.from_buffer_copy(blob)
    # read section headers
    shoff = header.e_shoff
    shnum = header.e_shnum
    shentsize = header.e_shentsize
    section_headers = []
    for i in range(shnum):
        off = shoff + i * shentsize
        sh = Elf64_Shdr.from_buffer_copy(blob[off:off+ctypes.sizeof(Elf64_Shdr)])
        section_headers.append(sh)
    # get shstrtab
    shstr_ndx = header.e_shstrndx
    if shstr_ndx >= len(section_headers):
        shstrtab = b""
    else:
        shsh = section_headers[shstr_ndx]
        shstrtab = blob[shsh.sh_offset: shsh.sh_offset + shsh.sh_size]
    # build sections
    sections = []
    for sh in section_headers:
        name = _strtab(shstrtab, sh.sh_name) if shstrtab else ""
        content = blob[sh.sh_offset: sh.sh_offset + sh.sh_size]
        sections.append(ElfSection(name, sh, content))
    # helper to create ctypes array from section content
    def _to_carray(sh: ElfSection, ctype):
        if sh.header.sh_entsize == 0:
            return []
        count = sh.header.sh_size // sh.header.sh_entsize
        arr_type = ctype * count
        return arr_type.from_buffer_copy(sh.content)
    # find rel/rela sections
    rels = []
    relas = []
    for sh in sections:
        if sh.header.sh_type == SHT_REL:
            rels.append((sh, sh.name[4:], _to_carray(sh, Elf64_Rel)))
        if sh.header.sh_type == SHT_RELA:
            # name like .rela.<target>
            relas.append((sh, sh.name[5:], _to_carray(sh, Elf64_Rela)))
    # symtab
    symtab_arr = []
    for sh in sections:
        if sh.header.sh_type == SHT_SYMTAB:
            arr = _to_carray(sh, Elf64_Sym)
            symtab_arr = list(arr)
            break
    progbits = [sh for sh in sections if sh.header.sh_type == SHT_PROGBITS]
    # Prealloc image for fixed addresses
    maxaddr = 0
    for sh in progbits:
        if sh.header.sh_addr != 0:
            maxaddr = max(maxaddr, sh.header.sh_addr + sh.header.sh_size)
    image = bytearray(maxaddr)
    # place progbits
    for sh in progbits:
        if sh.header.sh_addr != 0:
            start = sh.header.sh_addr
            image[start:start+len(sh.content)] = sh.content
        else:
            align = max(sh.header.sh_addralign or 1, force_section_align)
            pad = (-len(image)) % align
            if pad:
                image += b'\0' * pad
            sh.header.sh_addr = len(image)
            image += sh.content
    # collect relocations combining rel + rela
    reloc_entries = []
    for sh, target_name, c_rels in rels + relas:
        # find section that will be relocated (target_name is name after .rel. or .rela.)
        # But in original code, they used sh.name[4:] etc; we'll follow that idea:
        target_section = next((s for s in sections if s.name == target_name), None)
        if target_section is None:
            # fallback: target_name may be section name or symbol name; attempt best-effort
            continue
        target_image_off = target_section.header.sh_addr
        # build list of parsed rels
        parsed = []
        if isinstance(c_rels, list) or isinstance(c_rels, tuple):
            iterable = c_rels
        else:
            iterable = list(c_rels)
        for r in iterable:
            if isinstance(r, Elf64_Rela):
                r_offset = r.r_offset
                r_info = r.r_info
                r_addend = r.r_addend
            elif isinstance(r, Elf64_Rel):
                r_offset = r.r_offset
                r_info = r.r_info
                # addend is taken from location (we'll later fetch)
                r_addend = 0
            else:
                continue
            sym_idx = ELF64_R_SYM(r_info)
            r_type = ELF64_R_TYPE(r_info)
            parsed.append((r_offset, sym_idx, r_type, r_addend))
        # resolve symbol indices to sym objects
        for r_offset, sym_idx, r_type, r_addend in parsed:
            if sym_idx >= len(symtab_arr):
                # undefined or external symbol -> keep as is; will error later if unresolved
                sym = None
            else:
                sym = symtab_arr[sym_idx]
            reloc_entries.append((target_image_off + r_offset, sym, r_type, r_addend))
    return memoryview(image), sections, reloc_entries, symtab_arr

# -----------------------
# relocate implementation (arch-specific)
# -----------------------
def relocate(instr: int, ploc: int, tgt: int, r_type: int, arch: str) -> int:
    """
    instr: original 32-bit instruction value (as int)
    ploc: place location (address where instruction sits in final image)
    tgt:  target absolute address
    r_type: relocation type (int)
    arch: 'x86_64' or 'aarch64'
    """
    if arch.startswith('x86'):
        if r_type == R_X86_64_PC32:
            # PC-relative 32-bit: value = tgt - ploc
            return i2u(32, tgt - ploc)
        else:
            raise NotImplementedError(f"x86 relocation {r_type} not implemented")
    elif arch in ('aarch64', 'arm64'):
        # ARM64 cases from your earlier code
        if r_type == R_AARCH64_ADR_PREL_PG_HI21:
            rel_pg = (tgt & ~0xFFF) - (ploc & ~0xFFF)
            # bits: high 2 bits into instruction top, low 19 bits into middle (positions per earlier code)
            hi = getbits(rel_pg, 12, 14)    # bits 12..13 (2 bits)
            lo = getbits(rel_pg, 14, 32)    # bits 14..31 (18 bits?) adapt per encoding
            # ADR encoding placement (approx based on earlier code)
            return instr | (hi << 29) | (lo << 5)
        elif r_type == R_AARCH64_ADD_ABS_LO12_NC:
            return instr | (getbits(tgt, 0, 12) << 10)
        elif r_type == R_AARCH64_LDST16_ABS_LO12_NC:
            return instr | (getbits(tgt, 1, 12) << 10)
        elif r_type == R_AARCH64_LDST32_ABS_LO12_NC:
            return instr | (getbits(tgt, 2, 12) << 10)
        elif r_type == R_AARCH64_LDST64_ABS_LO12_NC:
            return instr | (getbits(tgt, 3, 12) << 10)
        elif r_type == R_AARCH64_LDST128_ABS_LO12_NC:
            return instr | (getbits(tgt, 4, 12) << 10)
        else:
            raise NotImplementedError(f"AArch64 relocation {r_type} not implemented")
    else:
        raise RuntimeError("Unknown arch for relocation")

# -----------------------
# jit_loader: apply relocations to image
# -----------------------
def jit_loader(obj: bytes, arch: str) -> Tuple[bytes, List[ElfSection], List[tuple], List[Elf64_Sym]]:
    image_mv, sections, relocs, symtab = elf_loader(obj)
    image = bytearray(image_mv)  # make mutable copy
    # For relocations we need to know where the target symbol is placed (symbol.st_shndx -> section index)
    # Build mapping section-index -> section object index in 'sections' list
    # In our toy loader, section order corresponds to section_headers ordering; sym.st_shndx refers to section index
    # We will assume sections list is in same order as original section headers.
    # Build a list of section addresses for index lookup:
    sect_addrs = [s.header.sh_addr for s in sections]
    # Now apply relocations
    processed_relocs = []
    for ploc, sym, r_type, r_addend in relocs:
        # read existing 4 or 8 bytes at ploc depending on arch; we assume 32-bit immediate fields (4 bytes)
        if ploc + 4 > len(image):
            raise RuntimeError("Relocation place outside image")
        old_val = struct.unpack_from("<I", image, ploc)[0]
        if sym is None:
            # undefined symbol -> cannot relocate
            name = "<UNDEF>"
            raise RuntimeError(f"Attempt to relocate undefined symbol at loc {ploc}")
        # compute symbol absolute address
        if sym.st_shndx == 0:
            raise RuntimeError(f'Attempting to relocate against an undefined symbol index {sym.st_name}')
        # sym.st_shndx indexes into section_headers ordering; find that section
        target_sh_idx = sym.st_shndx
        if target_sh_idx >= len(sections):
            raise RuntimeError("Symbol references invalid section index")
        target_section = sections[target_sh_idx]
        sym_addr = target_section.header.sh_addr + sym.st_value
        # compute final target with addend
        final_tgt = sym_addr + r_addend
        # call relocate to get new immediate / relocated value
        new_val = relocate(old_val, ploc, final_tgt, r_type, arch)
        # write back (we pack 4 bytes little endian)
        struct.pack_into("<I", image, ploc, new_val & 0xFFFFFFFF)
        processed_relocs.append((ploc, final_tgt, r_type))
    return bytes(image), sections, processed_relocs, symtab

# -----------------------
# ClangJITCompiler
# -----------------------
class ClangJITCompiler:
    def __init__(self, cachekey="compile_clang_jit"):
        self.cachekey = cachekey

    def compile(self, src: str) -> Tuple[bytes, str]:
        # choose target arch string (for clang --target)
        target_arch = 'x86_64' if sys.platform == 'win32' else platform.machine()
        # normalize arch string known by clang
        clang_target = 'x86_64' if target_arch.startswith('x86') else ('aarch64' if 'arm' in target_arch or 'aarch64' in target_arch else target_arch)
        args = ['-march=native', f'--target={clang_target}-none-unknown-elf', '-O2', '-fPIC', '-ffreestanding', '-fno-math-errno', '-nostdlib']
        arch_args = ['-ffixed-x18'] if clang_target == 'aarch64' and 'arm' in platform.machine() else []
        obj = subprocess.check_output([os.getenv("CC", 'clang'), '-c', '-x', 'c', *args, *arch_args, '-', '-o', '-'], input=src.encode('utf-8'))
        return obj, clang_target

    def disassemble(self, lib: bytes):
        # naive using objdump if available (best-effort)
        p = subprocess.Popen(['objdump', '-D', '-b', 'binary', '-m', 'i386:x86-64', '-Mintel'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(lib)
        return out.decode('utf-8', errors='replace')

# -----------------------
# Execution: map to memory, mprotect to RX, create ctypes function
# -----------------------
def make_executable_and_get_fn(binary: bytes, entry_offset: int = 0, restype=ctypes.c_int, argtypes=()):
    size = len(binary)
    # map RW
    mem = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANON, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    mem.write(binary)
    # address
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
    # mprotect to RX (must align to page)
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None)
    pagesize = mmap.PAGESIZE
    page_start = addr & ~(pagesize - 1)
    # length must cover the bytes from page_start to (addr+size)
    length = ((addr + size) - page_start + pagesize - 1) & ~(pagesize - 1)
    PROT_READ = mmap.PROT_READ
    PROT_EXEC = getattr(mmap, "PROT_EXEC", 0x4)
    res = libc.mprotect(ctypes.c_void_p(page_start), ctypes.c_size_t(length), PROT_READ | PROT_EXEC)
    if res != 0:
        raise OSError("mprotect failed, cannot make memory executable")
    # create function pointer starting at (addr + entry_offset)
    entry_addr = addr + entry_offset
    FUNC_TYPE = ctypes.CFUNCTYPE(restype, *argtypes)
    fn = FUNC_TYPE(entry_addr)
    # keep mem alive by attaching to function
    fn._mmap = mem
    return fn

# -----------------------
# Demo: compile a simple add() function and call it
# -----------------------
if __name__ == "__main__":
    # simple function; ensure no libc usage
    csrc = r'''
    int add(int a, int b) {
        return a + b;
    }
    '''
    print("Compiling with clang...")
    compiler = ClangJITCompiler()
    try:
        objbytes, arch = compiler.compile(csrc)
    except subprocess.CalledProcessError as e:
        print("clang failed:", e)
        sys.exit(1)

    print("Object size:", len(objbytes), "bytes. Running jit_loader...")
    try:
        image, sections, processed_relocs, symtab = jit_loader(objbytes, arch)
    except Exception as e:
        print("jit_loader failed:", e)
        sys.exit(1)

    print("Image length:", len(image))
    # Attempt to find symbol "add" in symtab
    sym_addr = None
    for sym in symtab:
        name = "<noname>"
        # find sym name by reading corresponding .strtab; attempt: find .strtab section
        # We try to locate section .strtab
        strtab_section = next((s for s in sections if s.name == ".strtab"), None)
        if strtab_section is not None:
            # get symbol name
            strtab_blob = strtab_section.content
            name = _strtab(strtab_blob, sym.st_name)
        else:
            # fallback: skip
            name = f"<idx{sym.st_name}>"
        if name == "add":
            # symbol's st_value is offset into its section; sym.st_shndx picks section index
            if sym.st_shndx == 0:
                continue
            sec = sections[sym.st_shndx]
            sym_addr = sec.header.sh_addr + sym.st_value
            break

    if sym_addr is None:
        # fallback: assume entry at offset 0
        print("Symbol 'add' not found in symtab; trying entry at offset 0")
        entry = 0
    else:
        entry = sym_addr

    print("Attempting to map image and call function at offset", entry)
    try:
        # create executable fn; entry is absolute address inside image, need offset
        # our make_executable expects entry_offset relative to image base (0)
        fn = make_executable_and_get_fn(image, entry_offset=entry, restype=ctypes.c_int, argtypes=(ctypes.c_int, ctypes.c_int))
        res = fn(10, 70)
        print("Result from JIT add(6,7):", res)
    except Exception as e:
        print("Execution failed:", e)
        sys.exit(1)
