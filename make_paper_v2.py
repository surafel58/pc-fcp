"""Tables and figures of the second manuscript version (paper_v2/, 2026-11-06) from the
ten-seed results (results/v5). Runs make_tables_paper.py and make_figures_paper.py with
their output folders redirected to paper_v2/ and three small source edits applied in
memory: "three seeds" becomes "ten seeds", the A8 row (tree-structured release) is part
of the ablation table, and the near-IID caption drops its old one-seed clause.
Usage (from dp-fcp-gate/): python make_paper_v2.py"""
import os
import sys

os.environ['PCFCP_RES'] = 'results/v5'
os.environ['PCFCP_EVAL'] = 'results/v5/eval'
os.environ['PCFCP_CACHE'] = 'results/v5/cache'
sys.path.insert(0, os.getcwd())


def patched(src, edits):
    code = open(src, encoding='utf-8').read()
    for old, new in edits:
        assert old in code, (src, old)
        code = code.replace(old, new)
    return code


tables = patched('make_tables_paper.py', [
    ("OUT = 'paper/tables'", "OUT = 'paper_v2/tables'"),
    ('three seeds', 'ten seeds'),
    ("('A7: equal width, $B = 50$', 'pcfcp_uniform50'),",
     "('A7: equal width, $B = 50$', 'pcfcp_uniform50'), ('A8: tree-structured release', 'abl_A8_tree'),"),
    ("r'bins of~\\eqref{eq:edges}.}',",
     "r'bins of~\\eqref{eq:edges}; A8 replaces the flat histogram by the tree-structured release '\n"
     "         r'of~\\cite{hay2010}.}',"),
    ("; the DP-FedCP rows are '", ". '"),
    ("r'from one seed, the others from three. $^\\dagger$", "r'$^\\dagger$"),
    ('assert len(a) == 3, cfg', 'assert len(a) == 10, cfg'),
    # the accuracy caption splits "three seeds" over two source lines
    ("standard deviation over three '", "standard deviation over ten '"),
    # every training configuration of the v5 runs has ten seeds, so the table lists all of them
    ("('cifar10_K10_dir100', 'CIFAR-10-LT', 10, '100'),",
     "('cifar10_K10_dir100', 'CIFAR-10-LT', 10, '100'), ('cifar10_K50_dir0.5', 'CIFAR-10-LT', 50, '0.5'), "
     "('cifar10_K100_dir0.5', 'CIFAR-10-LT', 100, '0.5'),"),
    ("('cifar100_K10_dir0.5', 'CIFAR-100-LT', 10, '0.5'),",
     "('cifar100_K10_dir0.5', 'CIFAR-100-LT', 10, '0.5'), "
     "('cifar100_K10_dir0.5_if100', 'CIFAR-100-LT, IF = 100', 10, '0.5'),"),
    # the v5 thesis tables (make_thesis_tables_v5.py) use Python formatting, so 85.05 prints as
    # 85.0 there; the same rule here keeps every paper number equal to the thesis number
    ("    return str(Decimal(repr(round(float(x), 10))).quantize(q, rounding=ROUND_HALF_UP))",
     "    return f'{float(x):.{nd}f}'"),
])
exec(compile(tables, 'make_tables_paper.py', 'exec'), {'__name__': '__main__'})

figures = patched('make_figures_paper.py', [("OUT = 'paper/figures'", "OUT = 'paper_v2/figures'")])
exec(compile(figures, 'make_figures_paper.py', 'exec'), {'__name__': '__main__'})
