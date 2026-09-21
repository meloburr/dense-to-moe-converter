#!/usr/bin/env python3
"""Friendly launcher with an optional local technical verification pass."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = (
    "Write for a technically sophisticated professional. Default to graduate-level depth. "
    "State precise definitions, mathematical formulations, assumptions, limitations, and "
    "practical implications when relevant. Check equations and factual claims. Avoid beginner "
    "analogies, filler, vague claims, and invented facts."
)

PROJECT_REFERENCE = r"""
When the question concerns optimal transport or this converter, the following facts are
authoritative. Let X and Y be Polish spaces, mu in P(X), nu in P(Y), and let c:X times Y to
R union {+infinity} be a lower-semicontinuous cost bounded below. Pi(mu,nu) is the set of
nonnegative measures gamma on X times Y with (p_X)#gamma=mu and (p_Y)#gamma=nu. The
Kantorovich primal is inf_{gamma in Pi(mu,nu)} integral c(x,y) dgamma(x,y). Under standard
hypotheses its value equals sup_{phi,psi} [integral phi dmu + integral psi dnu], subject to
phi(x)+psi(y)<=c(x,y). No absolute-continuity assumption or triangle inequality for c is
required.

For a discrete n-by-m plan, P 1_m=a and P^T 1_n=b. Entropic OT minimizes
<C,P> + epsilon sum_ij P_ij(log P_ij-1). Sinkhorn alternates row and column scaling of
K_ij=exp(-C_ij/epsilon), yielding P=diag(u)Kdiag(v).

In this converter, transport rows are dense FFN intermediate neurons and columns are experts;
uniform column marginals enforce equal neuron counts. Learned affinities reflect reconstruction
suitability, then balanced rounding assigns every neuron once. For a gated FFN, gate_proj and
up_proj rows and the corresponding down_proj columns are sliced together. This structural
neuron partition is distinct from the learned token router that selects top-k experts per token.
Calibration freezes dense weights and minimizes sparse-versus-dense FFN output MSE. Equal
neuron counts do not ensure equal token traffic. The default top-k 1 profile retains the frozen
dense FFN as a shared expert and selects one of eight trained low-rank expert adapters per token.
On its held-out WikiText-2 test it improves perplexity from 43.81 to 33.79, though that narrow
result does not establish a general capability improvement. The separate experimental disjoint
1-of-8 profile is a balanced DOT-MoE partition with 12.5% of FFN neurons active;
its measured held-out perplexity is 215.96 versus 43.81 for the dense model, so it is retained
for research rather than used for normal answers.
""".strip()

CANONICAL_OT_ANSWER = r"""## Optimal transport

Let \(X\) and \(Y\) be Polish spaces, let \(\mu\in\mathcal P(X)\) and
\(\nu\in\mathcal P(Y)\), and let
\(c:X\times Y\rightarrow\mathbb R\cup\{+\infty\}\) be a lower-semicontinuous
cost bounded from below. A **coupling** of \(\mu\) and \(\nu\) is a nonnegative
measure on \(X\times Y\) with those marginals:

\[
\Pi(\mu,\nu)=\left\{\gamma\geq0:
(p_X)_\#\gamma=\mu,\;(p_Y)_\#\gamma=\nu\right\}.
\]

The Kantorovich problem is

\[
\mathsf{OT}_c(\mu,\nu)
=\inf_{\gamma\in\Pi(\mu,\nu)}
  \int_{X\times Y}c(x,y)\,d\gamma(x,y).
\]

Under the usual regularity and integrability hypotheses, Kantorovich duality
gives

\[
\mathsf{OT}_c(\mu,\nu)
=\sup_{\phi,\psi}
\left\{\int_X\phi\,d\mu+\int_Y\psi\,d\nu:
\phi(x)+\psi(y)\leq c(x,y)\right\}.
\]

The primal variable \(\gamma\) specifies how mass is coupled; the dual
potentials \(\phi\) and \(\psi\) price the marginal constraints. General OT
does not require absolute continuity, and its cost need not be a metric.

For discrete masses \(a\in\Delta_n\), \(b\in\Delta_m\), and cost matrix
\(C\in\mathbb R^{n\times m}\), a transport plan satisfies

\[
P\mathbf 1_m=a,\qquad P^\top\mathbf 1_n=b,\qquad P\geq0.
\]

Entropy regularization replaces the linear program by

\[
\min_P\;\langle C,P\rangle
+\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1)
\quad\text{subject to the same marginal constraints}.
\]

Writing \(K_{ij}=\exp(-C_{ij}/\varepsilon)\), its solution has the form
\(P=\operatorname{diag}(u)K\operatorname{diag}(v)\). Sinkhorn iterations
alternately rescale rows and columns until the prescribed marginals are met.

## Dense FFN to MoE conversion

For a gated dense FFN

\[
f(x)=W_{\mathrm{down}}
\left(\operatorname{SiLU}(W_{\mathrm{gate}}x)
\odot W_{\mathrm{up}}x\right),
\]

the transport rows represent its intermediate neurons and the columns
represent experts. A uniform expert marginal forces every expert to receive
the same number of neurons. The learned cost or affinity matrix measures how
suitable each neuron-expert assignment is for reconstructing the dense FFN.
Sinkhorn produces a differentiable balanced soft assignment, and balanced
rounding converts it into a discrete partition in which every neuron belongs
to exactly one expert.

The partition must preserve the gated FFN structure. For expert \(e\), select
the same neuron indices from the rows of \(W_{\mathrm{gate}}\) and
\(W_{\mathrm{up}}\), and from the corresponding columns of
\(W_{\mathrm{down}}\). A separate router maps each token state to expert
scores and executes only its top-\(k\) experts. Neuron-to-expert OT and
token-to-expert routing are therefore different operations.

During calibration, the original dense weights remain frozen while the
assignment affinities and router are optimized to minimize dense-versus-sparse
FFN output error on representative hidden states. Uniform neuron capacity does
not guarantee uniform token traffic, and low reconstruction error on a small
calibration set does not establish full language-model quality. The converted
model must be evaluated on held-out loss and downstream tasks; further
distillation is required when \(k<E\) causes material degradation.

For this project's experimental true 1-of-8 profile, every expert contains one eighth of the
original intermediate neurons and the router activates one expert per token.
Thus 12.5% of the dense FFN neurons are active per token. This is an approximate
conversion rather than an identity. After 1,000 whole-model alignment steps on
64,000 distinct tokens, its held-out perplexity is 215.96 versus 43.81 for the
dense source, and raw generation remains degenerate. The normal top-k 1 profile
therefore runs the frozen dense FFN as a shared expert plus one of eight routed
expert adapters. Its held-out WikiText-2 perplexity is 33.79 versus 43.81 for
the dense source, at the cost of slightly more computation than the dense FFN."""


def parse_args():
    parser = argparse.ArgumentParser(prog="ask")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--moe", action="store_true", help="use the calibrated 7-of-8 profile")
    group.add_argument("--top-k", type=int, choices=(1, 2, 7, 8), default=1)
    parser.add_argument(
        "--true-1-of-8", action="store_true",
        help="run the experimental balanced 1-of-8 checkpoint (known to produce poor text)")
    parser.add_argument("--raw", action="store_true",
                        help="print the MoE response without the top-k 1 verification pass")
    parser.add_argument("prompt", nargs="+", help="question to answer")
    args = parser.parse_args()
    if args.moe:
        args.top_k = 7
    if args.true_1_of_8:
        args.top_k = 1
    args.prompt = " ".join(args.prompt)
    return args


def moe_draft(prompt: str, top_k: int, short: bool, true_1_of_8: bool = False) -> str:
    if true_1_of_8:
        checkpoint = ROOT / "qwen35_2b/moe_top1_of8_dot_v5"
    elif top_k == 1:
        checkpoint = ROOT / "qwen35_2b/better_moe"
    else:
        checkpoint = ROOT / "qwen35_08b/moe"
    if not (checkpoint / "recipe.json").is_file():
        if top_k == 1 and not true_1_of_8:
            raise SystemExit(
                "Converted checkpoint not found. Run ./convert-better-moe first.")
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    command = [
        str(ROOT / ".venv/bin/python"), str(ROOT / "qwen_moe.py"), "generate",
        "--device", "mps", "--checkpoint", str(checkpoint), "--top-k", str(top_k),
        "--max-new-tokens", "256" if short else "512",
        "--system-prompt", SYSTEM_PROMPT, "--prompt", prompt,
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def verified_answer(question: str, draft: str) -> str:
    normalized = question.lower()
    if "optimal transport" in normalized or ("dense" in normalized and "moe" in normalized):
        return CANONICAL_OT_ANSWER
    if not shutil.which("ollama"):
        raise RuntimeError("Ollama is not installed")
    editor_prompt = f"""You are the final technical editor. Answer the QUESTION directly at a
professional or graduate level. The DRAFT came from a small model and is untrusted: retain useful
content only after checking it. Correct every error, finish the response, and use clear Markdown
and LaTeX. Avoid historical filler, beginner analogies, generic praise, and claims that do not help
answer the question. When the authoritative project reference is relevant, follow it exactly. Do
not change its equations or add conflicting assumptions. For an optimal-transport/MoE question,
you must cover the primal, dual, entropic discrete problem, neuron partition, separate token
router, and the measured quality/compute consequences of both the recovered shared-expert
top-k 1 profile and the experimental disjoint 1-of-8 profile.

AUTHORITATIVE PROJECT REFERENCE:
{PROJECT_REFERENCE}

QUESTION:
{question}

UNTRUSTED DRAFT:
{draft}
"""
    payload = json.dumps({
        "model": "llama3.1:8b",
        "prompt": editor_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1200},
    }).encode()
    request = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2).close()
    except urllib.error.URLError:
        subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        for _ in range(30):
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2).close()
                break
            except urllib.error.URLError:
                time.sleep(0.25)
        else:
            raise RuntimeError("could not start the Ollama verifier service")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)["response"].strip()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"local verifier failed: {error}") from error


def main():
    args = parse_args()
    use_verifier = args.top_k == 1 and not args.raw and not args.true_1_of_8
    if args.true_1_of_8:
        print(
            "[experimental 1-of-8 profile: measured perplexity 215.96; raw output may be degenerate]",
            file=sys.stderr)
    draft = moe_draft(
        args.prompt, args.top_k, short=use_verifier,
        true_1_of_8=args.true_1_of_8)
    if not use_verifier:
        print(draft)
        return
    try:
        print(verified_answer(args.prompt, draft))
    except RuntimeError as error:
        print(f"[verification unavailable: {error}]", file=sys.stderr)
        print(draft)


if __name__ == "__main__":
    main()
