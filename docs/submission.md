# Submission checklist

Deadline: **Nov 11 2026, 5:00 PM EST**.

| # | Step | State |
|---|---|---|
| 1 | Public repo at a 40-char SHA | done — https://github.com/zvporter-ops/coldstart at `24848d56ba438ed5d3697d7279e003a104e21bed` |
| 2 | `hermes plugins validate` green at that SHA | done — passed at `24848d5` (security scan safe) |
| 3 | Catalog entry `plugin-catalog/coldstart.yaml` | PR open: https://github.com/NousResearch/hermes-agent/pull/150323 |
| 4 | Demo video, under 90s, plugin shown working | script in `docs/demo-script.md`; needs recording |
| 5 | Post on X tagging @NousResearch | pending |
| 6 | Submit both links at https://form.typeform.com/to/FwJLvOln | pending |

## X post draft

> Built Coldstart for the Hermes plugin hackathon: an evidence gate for any local AI
> deployment.
>
> Your own material outranks public material, public answers are labelled as public, and
> anything unsupported never reaches you. The model never grades itself.
>
> 90s demo 👇 @NousResearch

## Form fields

- **Plugin repo:** the public repo URL
- **Pull request:** the catalog PR URL
- **Demo video:** the X post URL
- **Name / contact:** Zachary Porter

## After the PR is open

The catalog checks run when a maintainer starts them; entries are reviewed every few days.
SHA bumps are separate PRs. Do not push changes to the plugin repo after pinning without
opening a SHA-bump PR, or the listing will drift from what was reviewed.
