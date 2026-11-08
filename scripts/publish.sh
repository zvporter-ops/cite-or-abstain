#!/usr/bin/env bash
# Publish Cite or Abstain and open the catalog PR.
#
# Prerequisite: `gh auth login` has succeeded (the script checks and stops if not).
# Everything else -- repo creation, owner pinning, SHA pinning, fork, branch, PR -- runs here.
set -euo pipefail

GH="${GH_BIN:-$HOME/.local/bin/gh}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if ! "$GH" auth status >/dev/null 2>&1; then
  echo "Not authenticated. Run:  $GH auth login" >&2
  exit 1
fi

USER="$("$GH" api user --jq .login)"
echo "Authenticated as $USER"

# ---------------------------------------------------------------- 1. own repo
if "$GH" repo view "$USER/cite-or-abstain" >/dev/null 2>&1; then
  echo "Repository $USER/cite-or-abstain already exists; pushing to it."
  git remote get-url origin >/dev/null 2>&1 || \
    git remote add origin "https://github.com/$USER/cite-or-abstain.git"
  git push -u origin HEAD
else
  echo "Creating public repository $USER/cite-or-abstain"
  "$GH" repo create cite-or-abstain --public --source=. --remote=origin --push \
    --description "Evidence-gated answering with a provenance hierarchy for any local AI deployment"
fi

# ------------------------------------------------- 2. pin the real owner, then the SHA
# The owner has to be correct in the files BEFORE the SHA is taken, because the SHA is what
# the catalog pins. Editing afterwards would pin a commit that no longer matches the tree.
if grep -q "github.com/zacharyporter/cite-or-abstain" plugin.yaml README.md docs/catalog-entry.yaml 2>/dev/null; then
  sed -i '' "s|https://github.com/zacharyporter/cite-or-abstain|https://github.com/$USER/cite-or-abstain|g" \
    plugin.yaml README.md docs/catalog-entry.yaml
  git add -A
  git commit -m "Pin repository owner to $USER"
  git push
fi

SHA="$(git rev-parse HEAD)"
echo "Pinned SHA: $SHA"

# ------------------------------------------------------------- 3. catalog entry
ENTRY="$(mktemp -t cite-or-abstain-catalog.XXXXXX.yaml)"
sed -e "s|https://github.com/zacharyporter/cite-or-abstain|https://github.com/$USER/cite-or-abstain|g" \
    -e "s|^sha: .*|sha: \"$SHA\"|" \
    -e "s|^maintainer: .*|maintainer: $USER|" \
    docs/catalog-entry.yaml > "$ENTRY"

# ------------------------------------------------------------------- 4. fork + PR
FORK_DIR="$(mktemp -d -t hermes-agent-fork.XXXXXX)"
echo "Forking NousResearch/hermes-agent (if needed) and cloning the fork"
"$GH" repo fork NousResearch/hermes-agent --clone=false >/dev/null 2>&1 || true
git clone --depth 1 "https://github.com/$USER/hermes-agent.git" "$FORK_DIR"
cd "$FORK_DIR"
git checkout -b add-cite-or-abstain
cp "$ENTRY" plugin-catalog/cite-or-abstain.yaml
git add plugin-catalog/cite-or-abstain.yaml
git -c user.email="${GIT_AUTHOR_EMAIL:-$USER@users.noreply.github.com}" \
    -c user.name="${GIT_AUTHOR_NAME:-$USER}" \
    commit -m "Add cite-or-abstain to the plugin catalog"
git push -u origin add-cite-or-abstain

BODY_FILE="$(mktemp -t pr-body.XXXXXX.md)"
cat > "$BODY_FILE" <<'BODY'
Adds `cite-or-abstain` v0.2.1, pinned at `__SHA__`.

**What it does.** An evidence gate for any AI deployment. Internal material (anything the
user feeds it) is retrieved and cited first; public reference material is used only when
internal material does not cover the question, and those answers are labelled as public;
when nothing covers the question the reply is refused rather than invented. Every reply is
then checked mechanically against the passages that turn actually retrieved: quoted spans
must be verbatim, and figures, dates, URLs and identifiers must appear in a retrieved
passage. No model is asked to grade its own answer. `/trust` scores the configured
deployment.

**Clearance rule, enforced on every path.** Private material never reaches a model that is
not local. The turn's `pre_llm_call` withholds it, and so do the plugin's own tools: with no
local `endpoint` set, `corpus_ask` answers from the public tier only (and reports how many
internal passages it withheld), while `corpus_audit` and `/trust` refuse rather than sample
internal passages into a remote model. The clearance notice only recommends `corpus_ask`
when a local endpoint exists to answer through.

**Validation.** `hermes plugins validate . --install-deps` passes at the pinned SHA
(manifest, loadable entry, declared tools/hooks match registrations, config schema, security
scan, desktop surface, no core override). 116 unit tests pass.

**Risky behaviour (rule 13).**
- Reads only the directories configured in `corpus_roots` / `public_roots`, plus its own
  index under `$HERMES_HOME/plugin-data/cite-or-abstain/`.
- Makes model calls through Hermes's configured model, or the OpenAI-compatible `endpoint`
  the user sets (default empty = the session model).
- Shells out to `pdftotext` / `tesseract` when installed, only on the user's own files.
- `COA_API_KEY` is optional and read through Hermes's secret scope (`agent.secret_scope`),
  never from a file.
- No telemetry, no self-update, no credential files.
- Registers `transform_llm_output` with `hold_stream=True`: the reply is withheld until the
  gate has run, so token streaming is off while the plugin is enabled. This is deliberate —
  the plugin exists to stop unchecked text reaching the user.

**Both halves in one package.** Agent side: 2 hooks, 3 tools, 1 command, 1 skill. Desktop
side: `desktop/plugin.js` (evidence pane + status-bar chip) over its own
`dashboard/plugin_api.py` backend at `/api/plugins/cite-or-abstain/`.
BODY

sed -i '' -e "s|__SHA__|$SHA|g" "$BODY_FILE"

"$GH" pr create --repo NousResearch/hermes-agent \
  --head "$USER:add-cite-or-abstain" \
  --title "Add cite-or-abstain to the plugin catalog" \
  --body-file "$BODY_FILE"

echo
echo "Done. PR opened. Next: record the demo (under 90s), post it on X tagging @NousResearch,"
echo "then submit both links at https://form.typeform.com/to/FwJLvOln"
