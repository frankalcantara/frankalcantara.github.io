# diagnostico.py
import sys

with open(sys.argv[1], 'rb') as f:
    raw = f.read()

# Verifica BOM
if raw.startswith(b'\xef\xbb\xbf'):
    print("⚠ PROBLEMA: Arquivo tem BOM (UTF-8 with BOM)")
    print("  Solução: Save with Encoding → UTF-8 (sem BOM)")
else:
    print("✓ Sem BOM")

# Verifica se começa com ---
if not raw.startswith(b'---'):
    print(f"⚠ PROBLEMA: Arquivo não começa com ---")
    print(f"  Primeiros bytes: {raw[:20]}")
else:
    print("✓ Começa com ---")

# Verifica line endings
if b'\r\n' in raw[:500]:
    print("⚠ Windows line endings (CRLF) - recomendado mudar para LF")
else:
    print("✓ Unix line endings (LF)")

# Verifica caracteres estranhos antes do ---
first_line = raw.split(b'\n')[0]
if first_line.strip() != b'---':
    print(f"⚠ PROBLEMA: Caracteres antes/depois do primeiro ---")
    print(f"  Hex: {first_line.hex()}")
else:
    print("✓ Primeiro --- está limpo")