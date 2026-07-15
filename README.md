# A2A healthcare walkthrough — local Google Cloud setup

This is a locally runnable version of the DeepLearning.AI A2A course demo. The
four A2A servers use Vertex AI models in your Google Cloud project:

| Server | Port | Model/service |
| --- | ---: | --- |
| Insurance policy agent | 9999 | Anthropic Claude Haiku 4.5 on Vertex AI |
| Health research agent | 9998 | Gemini 2.5 Flash with Google Search |
| Healthcare provider agent | 9997 | OpenAI gpt-oss 20B MaaS plus local MCP data |
| Healthcare orchestrator | 9996 | Gemini 2.5 Flash plus A2A handoffs |

Use Python 3.12 (the pinned course framework versions are not declared for
newer Python releases), [uv](https://docs.astral.sh/uv/), and the Google Cloud
CLI. `uv` can install the required Python automatically from `pyproject.toml`.

The root-level Python files are the runnable, assembled demo. The `L3` through
`L10` directories are the original progressive course notebooks. Their helper
imports now use the same local ADC implementation, but their embedded JupyterLab
terminal cells and `%%writefile` snapshots are course artifacts. Run the
root-level servers below; do not execute a notebook `%%writefile` cell over a
refactored root file.

## 1. Prepare a Google Cloud project

Use a Google Cloud project where billing is enabled. In the commands below,
replace `YOUR_GOOGLE_CLOUD_PROJECT_ID` with your own project ID.

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then
sign in and select the project:

```bash
gcloud auth login
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID
gcloud services enable aiplatform.googleapis.com \
  --project=YOUR_GOOGLE_CLOUD_PROJECT_ID
```

`aiplatform.googleapis.com` is the API used by this demo. Google currently calls
it the Agent Platform API in parts of the console and documentation; it is the
same service historically called the Vertex AI API.

### IAM roles

If you own the project, you probably already have enough administrative access.
For a separate developer identity, grant only what that identity needs:

- `roles/aiplatform.user` (Agent Platform / Vertex AI User): required by the
  account that runs the Python agents and sends model predictions.
- `roles/serviceusage.serviceUsageConsumer`: useful for a user-based ADC quota
  project and required if `gcloud auth application-default set-quota-project`
  reports a `serviceusage.services.use` denial.
- `roles/serviceusage.serviceUsageAdmin`: needed only by the administrator who
  enables APIs; it is not needed for normal agent execution.
- `roles/consumerprocurement.entitlementManager`: needed by the person who
  enables third-party/managed models and accepts their Model Garden terms. Do
  not grant this broadly to every runtime identity.

An administrator can grant the runtime roles to your Google user:

```bash
PROJECT_ID=YOUR_GOOGLE_CLOUD_PROJECT_ID
USER_EMAIL=your-google-account@example.com

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:$USER_EMAIL" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:$USER_EMAIL" \
  --role="roles/serviceusage.serviceUsageConsumer"
```

You can skip these commands when the account already has equivalent permissions
(for example through the project Owner role).

### Enable the non-Google models

In [Model Garden](https://console.cloud.google.com/vertex-ai/model-garden), with
your project selected:

1. Open **Claude Haiku 4.5**, click **Enable**, and accept Anthropic's terms.
2. Open **OpenAI gpt-oss 20B** with **API Service / MaaS**, click **Enable**, and
   accept any presented terms.

The configured Claude ID is `claude-haiku-4-5@20251001` and supports the global
endpoint. The configured gpt-oss ID is `gpt-oss-20b-maas` (the request namespace
is `openai/gpt-oss-20b-maas`) and is available in `us-central1`. Model names and
locations are environment variables so they can be changed if availability in
your account differs.

## 2. Authenticate this local application

Use Application Default Credentials (ADC). This is separate from `gcloud auth
login` and does not require a downloaded key:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project \
  YOUR_GOOGLE_CLOUD_PROJECT_ID
```

ADC reads the user credential stored by `gcloud` in its standard local location
and refreshes access tokens automatically. Do not commit downloaded key files or
project-specific credential files to this repository.

### Optional service-account authentication

No service-account key is required for local development. If you specifically
want to test as a service account, prefer keyless impersonation:

1. Create a service account and grant it `roles/aiplatform.user` on the project.
2. Grant your Google user `roles/iam.serviceAccountTokenCreator` on that service
   account.

For example, an administrator can perform steps 1 and 2 with:

```bash
PROJECT_ID=YOUR_GOOGLE_CLOUD_PROJECT_ID
USER_EMAIL=your-google-account@example.com
SERVICE_ACCOUNT=a2a-demo@$PROJECT_ID.iam.gserviceaccount.com

gcloud iam service-accounts create a2a-demo --project="$PROJECT_ID"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SERVICE_ACCOUNT" \
  --role="roles/aiplatform.user"
gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT" \
  --member="user:$USER_EMAIL" \
  --role="roles/iam.serviceAccountTokenCreator" \
  --project="$PROJECT_ID"
```

Then create impersonated ADC:

```bash
gcloud auth application-default login \
  --impersonate-service-account=a2a-demo@YOUR_GOOGLE_CLOUD_PROJECT_ID.iam.gserviceaccount.com
```

## 3. Configure and install the repository

From the repository root:

```bash
cp -n .env.example .env
uv sync --frozen
uv run python scripts/check_gcp_setup.py
```

Edit `.env` and replace `YOUR_GOOGLE_CLOUD_PROJECT_ID` with your project ID.
All four agents authenticate to Vertex AI with ADC.

The checker refreshes an OAuth token but hides its value. A successful result
confirms local authentication and project selection; model access is confirmed
when the corresponding server makes its first request.

## 4. Run the demo

Open four terminals in the repository root and start the dependent agents first:

```bash
# Terminal 1
uv run python a2a_policy_agent.py
```

```bash
# Terminal 2
uv run python a2a_research_agent.py
```

```bash
# Terminal 3
uv run python a2a_provider_agent.py
```

After all three report that they are listening, start the orchestrator:

```bash
# Terminal 4
uv run python a2a_healthcare_agent.py
```

The orchestrator is available at `http://127.0.0.1:9996`. Query it from a fifth
terminal:

```bash
uv run python scripts/query_healthcare_agent.py
```

You can pass a custom question as one quoted argument. Alternatively, use the
final client cell in `L10/L10.ipynb`; execute only its client/interaction cells
after the root-level servers have started.

## 5. Run the lesson notebooks

The notebooks in `L3` through `L10` are lesson material, so run them after the
local environment is configured and dependencies are installed:

```bash
uv run jupyter lab
```

Most lesson notebooks contain client cells that call an A2A server. Start the
matching server before running those cells, otherwise the notebook will fail to
connect.

| Lesson | Start before client cells |
| --- | --- |
| `L4`, `L5` | `uv run python a2a_policy_agent.py` |
| `L6` | `uv run python a2a_research_agent.py` |
| `L7` | `uv run python a2a_policy_agent.py` and `uv run python a2a_research_agent.py` |
| `L8`, `L9` | `uv run python a2a_provider_agent.py` |
| `L10` | Start the policy, research, provider, and healthcare servers from section 4 |

The `%%writefile` cells are course snapshots. Skip them when you are using the
refactored root-level files.

## Troubleshooting

- **ADC not found:** run `gcloud auth application-default login`; `gcloud auth
  login` alone is not enough for Python client libraries.
- **Wrong project or 403 quota-project error:** run the `set-quota-project`
  command above and confirm the ADC user has
  `roles/serviceusage.serviceUsageConsumer`.
- **`aiplatform.endpoints.predict` denied:** grant `roles/aiplatform.user` to the
  identity shown by `gcloud auth application-default login`.
- **Model not found / permission denied for Claude or gpt-oss:** enable that exact
  model in Model Garden, accept its terms, and keep the model's supported
  location from `.env.example`.
- **Port already in use:** change the matching port in `.env`. All clients read
  the same values.
- **Unexpected project error:** confirm `.env` uses your project ID and rerun
  `gcloud auth application-default set-quota-project`.

Official references: [local ADC setup](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment),
[ADC lookup order](https://cloud.google.com/docs/authentication/application-default-credentials),
[Vertex AI quickstart and IAM](https://cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart),
[Claude on Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude),
and [gpt-oss 20B model details](https://cloud.google.com/vertex-ai/generative-ai/docs/maas/openai/gpt-oss-20b).
