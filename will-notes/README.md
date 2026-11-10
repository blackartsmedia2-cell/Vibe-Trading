# Will's notes on this fork

This is Will's fork of [HKUDS/Vibe-Trading](https://github.com/HKUDS/Vibe-Trading), used as a reference and tool source for his TRADING project.

## House rules
- **Don't edit the original project's files.** Put everything of ours in `will-notes/` (or new files of our own) so the daily upstream sync never hits a conflict.
- Only one workflow is ours: `.github/workflows/will-daily-upstream-sync.yml`. It syncs `main` with HKUDS/Vibe-Trading every day at about 5 AM Pacific and switches off every other workflow.
- Anything taken from this repo goes through the TRADING project's `standing_protocol_strategy_testing.md` before it's trusted.
- **No live broker connection** without Will's explicit approval. Paper/shadow accounts first.

## Plan
1. Fork + daily sync + daily digest (done 2026-11-10).
2. Run Vibe-Trading as an MCP server on the trading PC and connect it to Claude (next).

## Checking for updates by hand
Actions tab → "Daily upstream sync (Will)" → Run workflow.
