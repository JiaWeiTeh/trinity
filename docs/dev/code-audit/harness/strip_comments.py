"""Blank every comment and docstring while preserving line numbers.

Lens A of the code audit reads source with all prose removed, so a wrong
docstring cannot tell the reader what to see. Line numbers are preserved
exactly (prose is replaced by blanks, never deleted) so Lens A can cite
`file.py:line` against the *original* file.

    python docs/dev/code-audit/harness/strip_comments.py trinity <outdir>

Writes a mirrored tree of stripped copies under <outdir>.
"""

import ast
import pathlib
import sys
import tokenize


def _blank(lines, srow, scol, erow, ecol):
    """Replace the (srow,scol)-(erow,ecol) span with spaces, keeping line count."""
    if srow == erow:
        line = lines[srow - 1]
        lines[srow - 1] = line[:scol] + " " * (ecol - scol) + line[ecol:]
        return
    lines[srow - 1] = lines[srow - 1][:scol]
    for row in range(srow, erow - 1):
        lines[row] = ""
    lines[erow - 1] = " " * ecol + lines[erow - 1][ecol:]


def strip(source):
    lines = source.splitlines()

    # Comments run to end of line, so blanking them cannot disturb any span
    # that starts earlier on the same line (docstrings included).
    readline = iter(source.splitlines(keepends=True)).__next__
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.COMMENT:
            _blank(lines, tok.start[0], tok.start[1], tok.end[0], tok.end[1])

    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = node.body
        if not body or not isinstance(body[0], ast.Expr):
            continue
        doc = body[0].value
        if isinstance(doc, ast.Constant) and isinstance(doc.value, str):
            # Leave a bare `...` behind so a function whose whole body was a
            # docstring stays syntactically valid.
            _blank(lines, doc.lineno, doc.col_offset, doc.end_lineno, doc.end_col_offset)
            lines[doc.lineno - 1] = (
                lines[doc.lineno - 1][: doc.col_offset]
                + "..."
                + lines[doc.lineno - 1][doc.col_offset + 3 :]
            )

    return "\n".join(line.rstrip() for line in lines) + "\n"


def main(src_root, out_root):
    src_root, out_root = pathlib.Path(src_root), pathlib.Path(out_root)
    n = 0
    for path in sorted(src_root.rglob("*.py")):
        out = out_root / path.relative_to(src_root.parent)
        out.parent.mkdir(parents=True, exist_ok=True)
        stripped = strip(path.read_text())
        ast.parse(stripped)  # stripping must not break the file
        out.write_text(stripped)
        n += 1
    print(f"stripped {n} files -> {out_root}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
