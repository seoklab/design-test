# Protein Design Competition Platform

A reusable platform for running protein design competitions using AlphaFold3 structure prediction.

## Features

- Web-based sequence submission form
- Automatic validation and job queuing
- AlphaFold3 structure prediction on HPC/SLURM
- Private results with token-based URLs
- Email notifications (submission received + results ready)
- Interactive 3D structure viewer (Mol*/PDBe-Molstar)

## Quick Start (For Your Own Competition)

### Prerequisites

- GitHub repository (can be under an organization)
- Self-hosted GitHub Actions runner on HPC with:
  - SLURM job scheduler
  - AlphaFold3 installed
  - `sendmail` configured
  - Python 3.8+
- Netlify account (free tier works)

### Step 1: Clone and Configure

```bash
git clone https://github.com/seoklab/design-test.git my-competition
cd my-competition
```

### Step 2: Update Paths

Search and replace these paths throughout the codebase:

| Find | Replace With |
|------|--------------|
| `/data/galaxy4/user/j2ho/kidds2026/protein-competition` | Your base directory |
| `/data/galaxy4/user/j2ho/job_queue` | Your job queue directory |
| `seoklab.github.io/design-test` | Your GitHub Pages URL |
| `noreply@seoklab.org` | Your email sender address |

**Files to update:**
- `.github/workflows/process_submission.yml` - submission paths
- `.github/workflows/check_completion.yml` - paths and SITE_URL
- `scripts/run_af3.py` - AF3 paths and SLURM settings

### Step 3: Configure GitHub

1. **Enable GitHub Pages**: Settings → Pages → Source: `main` branch, `/docs` folder
2. **Set up self-hosted runner**: Settings → Actions → Runners → New self-hosted runner
3. **Update runner labels** in workflow files to match your runner

### Step 4: Configure Netlify

1. Create new site from Git
2. Set environment variables:
   - `GITHUB_TOKEN`: Personal access token with `repo` scope
   - `GITHUB_OWNER`: Your GitHub username or org
   - `GITHUB_REPO`: Your repository name
3. Deploy the site
4. Update `SUBMIT_URL` in `docs/index.html` with your Netlify URL

### Step 5: Set Up Job Queue

Create a cron job on your HPC to submit queued jobs:

```bash
# Example crontab entry (runs every 15 minutes)
*/15 * * * * /path/to/submit_queued_jobs.sh
```

Example `submit_queued_jobs.sh`:
```bash
#!/bin/bash
QUEUE_DIR="/your/job_queue"
for script in "$QUEUE_DIR"/*.sh; do
    [ -f "$script" ] && sbatch "$script" && mv "$script" "$script.submitted"
done
```

### Step 6: Customize Branding

Edit `docs/index.html`:
- Page title and header
- Competition info (goal, dates, deadline)
- Footer credits

## Configuration Reference

### Workflow Files

**`.github/workflows/process_submission.yml`**
- `SUBMISSION_DIR`: Where submissions are stored
- `QUEUE_DIR`: Where SLURM scripts are queued
- Runner labels: `[self-hosted, your-runner]`

**`.github/workflows/check_completion.yml`**
- `SUBMISSIONS_BASE`: Base path for submissions
- `PUBLIC_RESULTS`: Where packaged results go
- `SITE_URL`: Your GitHub Pages URL
- Schedule: Adjust cron or disable with comments

### Scripts

**`scripts/run_af3.py`**
- AF3 installation path
- SLURM partition and resources
- GPU configuration

**`scripts/package_results.py`**
- Files to include in results package

### Netlify Function

**`netlify/functions/submit.js`**
- Sequence length limits
- Validation rules

## Architecture

```
User → Web Form → Netlify Function → GitHub Issue
                                          ↓
                                   Process Submission
                                   (GitHub Actions)
                                          ↓
                                   SLURM Job Queue
                                          ↓
                                   AlphaFold3 (HPC)
                                          ↓
                                   Check Completion
                                   (GitHub Actions)
                                          ↓
                         Email + Viewer Link → User
```

## File Structure

```
├── .github/workflows/
│   ├── process_submission.yml    # Handles new submissions
│   └── check_completion.yml      # Checks for completed jobs
├── docs/                         # GitHub Pages
│   ├── index.html                # Submission form
│   ├── viewer.html               # Mol* structure viewer
│   ├── SYSTEM_OVERVIEW.md        # Detailed documentation
│   └── results/                  # Packaged results (token-based)
├── netlify/functions/
│   └── submit.js                 # Form submission API
├── scripts/
│   ├── parse_submission.py       # Parse issue → submission.json
│   ├── prepare_af3_input.py      # Generate AF3 input JSON
│   ├── run_af3.py                # Generate SLURM script
│   └── package_results.py        # Package results with token
├── submissions/                  # (gitignored) Active submissions
└── public_results/               # (gitignored) Packaged results
```

## Sequence Requirements

Default settings (configurable in `netlify/functions/submit.js`):
- **Length**: 10 - 5,000 residues
- **Valid amino acids**: A C D E F G H I K L M N P Q R S T V W Y

## Maintenance

### Enable/Disable Scheduled Checks

```yaml
# In .github/workflows/check_completion.yml
on:
  # Comment out to disable
  schedule:
    - cron: '* * * * *'
  workflow_dispatch:  # Always keep for manual triggers
```

### Manual Triggers

- **Actions** → **Check Job Completion** → **Run workflow**

### Deploy Changes

- **GitHub Pages** (docs/): Auto-deploys on push
- **Netlify Function**: Requires manual redeploy or re-link repo

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Git push rejected | `git pull --rebase && git push` |
| Permission denied on files | Check file ownership between users |
| Form submission fails | Check Netlify function logs and env vars |
| Emails not sending | Verify sendmail configuration on HPC |
| Viewer not loading | Check browser console for CORS/URL errors |

## License

MIT License - Feel free to use and modify for your own competitions.

## Credits

- [AlphaFold3](https://github.com/google-deepmind/alphafold3) - Structure prediction
- [PDBe-Molstar](https://github.com/molstar/pdbe-molstar) - 3D visualization
- Original implementation by [SeokLab](https://seoklab.org)
