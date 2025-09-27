+ sudo mkdir /app
+ sudo chown 1001 /app
+ git config --global core.hooksPath /dev/null
+ git config --global --add url.http://git@192.168.0.1:8080/.insteadOf https://github.com/
+ git config --global --add url.http://git@192.168.0.1:8080/.insteadOf git@github.com:
+ git clone --depth 1 --shallow-submodules --recurse-submodules https://github.com/FMProducer/prosperous_bot /app
Cloning into '/app'...
--- Starting Initial Setup ---
+ cd /app
+ set -e
+ echo '--- Starting Initial Setup ---'
+ echo '--- Installing Python dependencies ---'
--- Installing Python dependencies ---
+ pip install -r requirements.txt
Collecting aiohttp (from -r requirements.txt (line 1))
  Downloading aiohttp-3.12.15-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.7 kB)
Collecting fastapi (from -r requirements.txt (line 2))
  Downloading fastapi-0.116.2-py3-none-any.whl.metadata (28 kB)
Collecting gate-api (from -r requirements.txt (line 3))
  Downloading gate_api-7.1.8-py3-none-any.whl.metadata (62 kB)
Collecting hypothesis (from -r requirements.txt (line 4))
  Downloading hypothesis-6.139.1-py3-none-any.whl.metadata (5.6 kB)
Collecting jinja2 (from -r requirements.txt (line 5))
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting joblib (from -r requirements.txt (line 6))
  Downloading joblib-1.5.2-py3-none-any.whl.metadata (5.6 kB)
Collecting matplotlib (from -r requirements.txt (line 7))
  Downloading matplotlib-3.10.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting numpy (from -r requirements.txt (line 8))
  Downloading numpy-2.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (62 kB)
Collecting optuna (from -r requirements.txt (line 9))
  Downloading optuna-4.5.0-py3-none-any.whl.metadata (17 kB)
Collecting pandas (from -r requirements.txt (line 10))
  Downloading pandas-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (91 kB)
Collecting plotly (from -r requirements.txt (line 11))
  Downloading plotly-6.3.0-py3-none-any.whl.metadata (8.5 kB)
Collecting pytest (from -r requirements.txt (line 12))
  Downloading pytest-8.4.2-py3-none-any.whl.metadata (7.7 kB)
Collecting pytest-asyncio (from -r requirements.txt (line 13))
  Downloading pytest_asyncio-1.2.0-py3-none-any.whl.metadata (4.1 kB)
Collecting requests (from -r requirements.txt (line 14))
  Downloading requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting scikit-learn (from -r requirements.txt (line 15))
  Downloading scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
Collecting ta (from -r requirements.txt (line 16))
  Downloading ta-0.11.0.tar.gz (25 kB)
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Getting requirements to build wheel: started
  Getting requirements to build wheel: finished with status 'done'
  Preparing metadata (pyproject.toml): started
  Preparing metadata (pyproject.toml): finished with status 'done'
Collecting uvicorn (from -r requirements.txt (line 17))
  Downloading uvicorn-0.35.0-py3-none-any.whl.metadata (6.5 kB)
Collecting xgboost (from -r requirements.txt (line 18))
  Downloading xgboost-3.0.5-py3-none-manylinux_2_28_x86_64.whl.metadata (2.1 kB)
Collecting pytest-cov (from -r requirements.txt (line 19))
  Downloading pytest_cov-7.0.0-py3-none-any.whl.metadata (31 kB)
Collecting pytest-mock (from -r requirements.txt (line 20))
  Downloading pytest_mock-3.15.1-py3-none-any.whl.metadata (3.9 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp->-r requirements.txt (line 1))
  Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp->-r requirements.txt (line 1))
  Downloading aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp->-r requirements.txt (line 1))
  Downloading attrs-25.3.0-py3-none-any.whl.metadata (10 kB)
Collecting frozenlist>=1.1.1 (from aiohttp->-r requirements.txt (line 1))
  Downloading frozenlist-1.7.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp->-r requirements.txt (line 1))
  Downloading multidict-6.6.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
Collecting propcache>=0.2.0 (from aiohttp->-r requirements.txt (line 1))
  Downloading propcache-0.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (12 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp->-r requirements.txt (line 1))
  Downloading yarl-1.20.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (73 kB)
Collecting idna>=2.0 (from yarl<2.0,>=1.17.0->aiohttp->-r requirements.txt (line 1))
  Using cached idna-3.10-py3-none-any.whl.metadata (10 kB)
Collecting starlette<0.49.0,>=0.40.0 (from fastapi->-r requirements.txt (line 2))
  Downloading starlette-0.48.0-py3-none-any.whl.metadata (6.3 kB)
Collecting pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4 (from fastapi->-r requirements.txt (line 2))
  Downloading pydantic-2.11.9-py3-none-any.whl.metadata (68 kB)
Requirement already satisfied: typing-extensions>=4.8.0 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from fastapi->-r requirements.txt (line 2)) (4.14.1)
Collecting annotated-types>=0.6.0 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi->-r requirements.txt (line 2))
  Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.33.2 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi->-r requirements.txt (line 2))
  Downloading pydantic_core-2.33.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
Collecting typing-inspection>=0.4.0 (from pydantic!=1.8,!=1.8.1,!=2.0.0,!=2.0.1,!=2.1.0,<3.0.0,>=1.7.4->fastapi->-r requirements.txt (line 2))
  Downloading typing_inspection-0.4.1-py3-none-any.whl.metadata (2.6 kB)
Collecting anyio<5,>=3.6.2 (from starlette<0.49.0,>=0.40.0->fastapi->-r requirements.txt (line 2))
  Downloading anyio-4.10.0-py3-none-any.whl.metadata (4.0 kB)
Collecting sniffio>=1.1 (from anyio<5,>=3.6.2->starlette<0.49.0,>=0.40.0->fastapi->-r requirements.txt (line 2))
  Using cached sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
Collecting urllib3>=1.15 (from gate-api->-r requirements.txt (line 3))
  Using cached urllib3-2.5.0-py3-none-any.whl.metadata (6.5 kB)
Collecting six>=1.10 (from gate-api->-r requirements.txt (line 3))
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting certifi (from gate-api->-r requirements.txt (line 3))
  Downloading certifi-2025.8.3-py3-none-any.whl.metadata (2.4 kB)
Collecting python-dateutil (from gate-api->-r requirements.txt (line 3))
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting sortedcontainers<3.0.0,>=2.1.0 (from hypothesis->-r requirements.txt (line 4))
  Downloading sortedcontainers-2.4.0-py2.py3-none-any.whl.metadata (10 kB)
Collecting MarkupSafe>=2.0 (from jinja2->-r requirements.txt (line 5))
  Using cached MarkupSafe-3.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.0 kB)
Collecting contourpy>=1.0.1 (from matplotlib->-r requirements.txt (line 7))
  Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib->-r requirements.txt (line 7))
  Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib->-r requirements.txt (line 7))
  Downloading fonttools-4.60.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl.metadata (111 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib->-r requirements.txt (line 7))
  Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (6.3 kB)
Collecting packaging>=20.0 (from matplotlib->-r requirements.txt (line 7))
  Using cached packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pillow>=8 (from matplotlib->-r requirements.txt (line 7))
  Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (9.0 kB)
Collecting pyparsing>=2.3.1 (from matplotlib->-r requirements.txt (line 7))
  Downloading pyparsing-3.2.4-py3-none-any.whl.metadata (5.0 kB)
Collecting alembic>=1.5.0 (from optuna->-r requirements.txt (line 9))
  Downloading alembic-1.16.5-py3-none-any.whl.metadata (7.3 kB)
Collecting colorlog (from optuna->-r requirements.txt (line 9))
  Downloading colorlog-6.9.0-py3-none-any.whl.metadata (10 kB)
Collecting sqlalchemy>=1.4.2 (from optuna->-r requirements.txt (line 9))
  Downloading sqlalchemy-2.0.43-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.6 kB)
Collecting tqdm (from optuna->-r requirements.txt (line 9))
  Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
Collecting PyYAML (from optuna->-r requirements.txt (line 9))
  Using cached PyYAML-6.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.1 kB)
Collecting pytz>=2020.1 (from pandas->-r requirements.txt (line 10))
  Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas->-r requirements.txt (line 10))
  Downloading tzdata-2025.2-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting narwhals>=1.15.1 (from plotly->-r requirements.txt (line 11))
  Downloading narwhals-2.5.0-py3-none-any.whl.metadata (11 kB)
Collecting iniconfig>=1 (from pytest->-r requirements.txt (line 12))
  Using cached iniconfig-2.1.0-py3-none-any.whl.metadata (2.7 kB)
Collecting pluggy<2,>=1.5 (from pytest->-r requirements.txt (line 12))
  Using cached pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
Collecting pygments>=2.7.2 (from pytest->-r requirements.txt (line 12))
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting charset_normalizer<4,>=2 (from requests->-r requirements.txt (line 14))
  Downloading charset_normalizer-3.4.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (36 kB)
Collecting scipy>=1.8.0 (from scikit-learn->-r requirements.txt (line 15))
  Downloading scipy-1.16.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (62 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn->-r requirements.txt (line 15))
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting click>=7.0 (from uvicorn->-r requirements.txt (line 17))
  Using cached click-8.2.1-py3-none-any.whl.metadata (2.5 kB)
Collecting h11>=0.8 (from uvicorn->-r requirements.txt (line 17))
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Collecting nvidia-nccl-cu12 (from xgboost->-r requirements.txt (line 18))
  Downloading nvidia_nccl_cu12-2.28.3-py3-none-manylinux_2_18_x86_64.whl.metadata (2.0 kB)
Collecting coverage>=7.10.6 (from coverage[toml]>=7.10.6->pytest-cov->-r requirements.txt (line 19))
  Downloading coverage-7.10.6-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (8.9 kB)
Collecting Mako (from alembic>=1.5.0->optuna->-r requirements.txt (line 9))
  Downloading mako-1.3.10-py3-none-any.whl.metadata (2.9 kB)
Requirement already satisfied: greenlet>=1 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from sqlalchemy>=1.4.2->optuna->-r requirements.txt (line 9)) (3.2.3)
Downloading aiohttp-3.12.15-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 29.9 MB/s eta 0:00:00
Downloading multidict-6.6.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (256 kB)
Downloading yarl-1.20.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (355 kB)
Downloading fastapi-0.116.2-py3-none-any.whl (95 kB)
Downloading pydantic-2.11.9-py3-none-any.whl (444 kB)
Downloading pydantic_core-2.33.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 33.8 MB/s eta 0:00:00
Downloading starlette-0.48.0-py3-none-any.whl (73 kB)
Downloading anyio-4.10.0-py3-none-any.whl (107 kB)
Downloading gate_api-7.1.8-py3-none-any.whl (640 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 640.8/640.8 kB 45.0 MB/s eta 0:00:00
Downloading hypothesis-6.139.1-py3-none-any.whl (533 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 533.8/533.8 kB 20.5 MB/s eta 0:00:00
Downloading sortedcontainers-2.4.0-py2.py3-none-any.whl (29 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading joblib-1.5.2-py3-none-any.whl (308 kB)
Downloading matplotlib-3.10.6-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 36.0 MB/s eta 0:00:00
Downloading numpy-2.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (16.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.6/16.6 MB 47.2 MB/s eta 0:00:00
Downloading optuna-4.5.0-py3-none-any.whl (400 kB)
Downloading pandas-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 59.1 MB/s eta 0:00:00
Downloading plotly-6.3.0-py3-none-any.whl (9.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.8/9.8 MB 53.0 MB/s eta 0:00:00
Downloading pytest-8.4.2-py3-none-any.whl (365 kB)
Using cached pluggy-1.6.0-py3-none-any.whl (20 kB)
Downloading pytest_asyncio-1.2.0-py3-none-any.whl (15 kB)
Downloading requests-2.32.5-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (151 kB)
Using cached idna-3.10-py3-none-any.whl (70 kB)
Using cached urllib3-2.5.0-py3-none-any.whl (129 kB)
Downloading scikit_learn-1.7.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.5/9.5 MB 37.5 MB/s eta 0:00:00
Downloading uvicorn-0.35.0-py3-none-any.whl (66 kB)
Downloading xgboost-3.0.5-py3-none-manylinux_2_28_x86_64.whl (94.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 94.9/94.9 MB 38.0 MB/s eta 0:00:00
Downloading pytest_cov-7.0.0-py3-none-any.whl (22 kB)
Downloading pytest_mock-3.15.1-py3-none-any.whl (10 kB)
Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl (15 kB)
Downloading aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Downloading alembic-1.16.5-py3-none-any.whl (247 kB)
Downloading annotated_types-0.7.0-py3-none-any.whl (13 kB)
Downloading attrs-25.3.0-py3-none-any.whl (63 kB)
Downloading certifi-2025.8.3-py3-none-any.whl (161 kB)
Using cached click-8.2.1-py3-none-any.whl (102 kB)
Downloading contourpy-1.3.3-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (362 kB)
Downloading coverage-7.10.6-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (251 kB)
Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.60.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl (4.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.9/4.9 MB 146.9 MB/s eta 0:00:00
Downloading frozenlist-1.7.0-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (241 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Using cached iniconfig-2.1.0-py3-none-any.whl (6.0 kB)
Downloading kiwisolver-1.4.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 119.6 MB/s eta 0:00:00
Using cached MarkupSafe-3.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (23 kB)
Downloading narwhals-2.5.0-py3-none-any.whl (407 kB)
Using cached packaging-25.0-py3-none-any.whl (66 kB)
Downloading pillow-11.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.6/6.6 MB 164.9 MB/s eta 0:00:00
Downloading propcache-0.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (224 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Downloading pyparsing-3.2.4-py3-none-any.whl (113 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pytz-2025.2-py2.py3-none-any.whl (509 kB)
Downloading scipy-1.16.2-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (35.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 35.7/35.7 MB 79.1 MB/s eta 0:00:00
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
Downloading sqlalchemy-2.0.43-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 156.1 MB/s eta 0:00:00
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading typing_inspection-0.4.1-py3-none-any.whl (14 kB)
Downloading tzdata-2025.2-py2.py3-none-any.whl (347 kB)
Downloading colorlog-6.9.0-py3-none-any.whl (11 kB)
Downloading mako-1.3.10-py3-none-any.whl (78 kB)
Downloading nvidia_nccl_cu12-2.28.3-py3-none-manylinux_2_18_x86_64.whl (295.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 295.9/295.9 MB 39.1 MB/s eta 0:00:00
Using cached PyYAML-6.0.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (767 kB)
Downloading tqdm-4.67.1-py3-none-any.whl (78 kB)
Building wheels for collected packages: ta
  Building wheel for ta (pyproject.toml): started
  Building wheel for ta (pyproject.toml): finished with status 'done'
  Created wheel for ta: filename=ta-0.11.0-py3-none-any.whl size=29482 sha256=c8ca4f1e82d381bb23f8712458eee7d4bf879a7bd37fef7f65c4975e176e9d97
  Stored in directory: /home/jules/.cache/pip/wheels/5c/a1/5f/c6b85a7d9452057be4ce68a8e45d77ba34234a6d46581777c6
Successfully built ta
Installing collected packages: sortedcontainers, pytz, urllib3, tzdata, typing-inspection, tqdm, threadpoolctl, sqlalchemy, sniffio, six, PyYAML, pyparsing, pygments, pydantic-core, propcache, pluggy, pillow, packaging, nvidia-nccl-cu12, numpy, narwhals, multidict, MarkupSafe, kiwisolver, joblib, iniconfig, idna, h11, frozenlist, fonttools, cycler, coverage, colorlog, click, charset_normalizer, certifi, attrs, annotated-types, aiohappyeyeballs, yarl, uvicorn, scipy, requests, python-dateutil, pytest, pydantic, plotly, Mako, jinja2, hypothesis, contourpy, anyio, aiosignal, xgboost, starlette, scikit-learn, pytest-mock, pytest-cov, pytest-asyncio, pandas, matplotlib, gate-api, alembic, aiohttp, ta, optuna, fastapi

Successfully installed Mako-1.3.10 MarkupSafe-3.0.2 PyYAML-6.0.2 aiohappyeyeballs-2.6.1 aiohttp-3.12.15 aiosignal-1.4.0 alembic-1.16.5 annotated-types-0.7.0 anyio-4.10.0 attrs-25.3.0 certifi-2025.8.3 charset_normalizer-3.4.3 click-8.2.1 colorlog-6.9.0 contourpy-1.3.3 coverage-7.10.6 cycler-0.12.1 fastapi-0.116.2 fonttools-4.60.0 frozenlist-1.7.0 gate-api-7.1.8 h11-0.16.0 hypothesis-6.139.1 idna-3.10 iniconfig-2.1.0 jinja2-3.1.6 joblib-1.5.2 kiwisolver-1.4.9 matplotlib-3.10.6 multidict-6.6.4 narwhals-2.5.0 numpy-2.3.3 nvidia-nccl-cu12-2.28.3 optuna-4.5.0 packaging-25.0 pandas-2.3.2 pillow-11.3.0 plotly-6.3.0 pluggy-1.6.0 propcache-0.3.2 pydantic-2.11.9 pydantic-core-2.33.2 pygments-2.19.2 pyparsing-3.2.4 pytest-8.4.2 pytest-asyncio-1.2.0 pytest-cov-7.0.0 pytest-mock-3.15.1 python-dateutil-2.9.0.post0 pytz-2025.2 requests-2.32.5 scikit-learn-1.7.2 scipy-1.16.2 six-1.17.0 sniffio-1.3.1 sortedcontainers-2.4.0 sqlalchemy-2.0.43 starlette-0.48.0 ta-0.11.0 threadpoolctl-3.6.0 tqdm-4.67.1 typing-inspection-0.4.1 tzdata-2025.2 urllib3-2.5.0 uvicorn-0.35.0 xgboost-3.0.5 yarl-1.20.1

[notice] A new release of pip is available: 25.1.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
--- Installing project in editable mode ---
+ echo '--- Installing project in editable mode ---'
+ pip install -e .
Obtaining file:///app
  Installing build dependencies: started
  Installing build dependencies: finished with status 'done'
  Checking if build backend supports build_editable: started
  Checking if build backend supports build_editable: finished with status 'done'
  Getting requirements to build editable: started
  Getting requirements to build editable: finished with status 'done'
  Preparing editable metadata (pyproject.toml): started
  Preparing editable metadata (pyproject.toml): finished with status 'done'
Requirement already satisfied: gate-api>=6.27 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from prosperous_bot==0.1.0) (7.1.8)
Requirement already satisfied: pytest in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from prosperous_bot==0.1.0) (8.4.2)
Requirement already satisfied: pytest-asyncio in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from prosperous_bot==0.1.0) (1.2.0)
Requirement already satisfied: hypothesis in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from prosperous_bot==0.1.0) (6.139.1)
Requirement already satisfied: urllib3>=1.15 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from gate-api>=6.27->prosperous_bot==0.1.0) (2.5.0)
Requirement already satisfied: six>=1.10 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from gate-api>=6.27->prosperous_bot==0.1.0) (1.17.0)
Requirement already satisfied: certifi in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from gate-api>=6.27->prosperous_bot==0.1.0) (2025.8.3)
Requirement already satisfied: python-dateutil in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from gate-api>=6.27->prosperous_bot==0.1.0) (2.9.0.post0)
Requirement already satisfied: attrs>=22.2.0 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from hypothesis->prosperous_bot==0.1.0) (25.3.0)
Requirement already satisfied: sortedcontainers<3.0.0,>=2.1.0 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from hypothesis->prosperous_bot==0.1.0) (2.4.0)
Requirement already satisfied: iniconfig>=1 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from pytest->prosperous_bot==0.1.0) (2.1.0)
Requirement already satisfied: packaging>=20 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from pytest->prosperous_bot==0.1.0) (25.0)
Requirement already satisfied: pluggy<2,>=1.5 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from pytest->prosperous_bot==0.1.0) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from pytest->prosperous_bot==0.1.0) (2.19.2)
Requirement already satisfied: typing-extensions>=4.12 in /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages (from pytest-asyncio->prosperous_bot==0.1.0) (4.14.1)
Building wheels for collected packages: prosperous_bot
  Building editable for prosperous_bot (pyproject.toml): started
  Building editable for prosperous_bot (pyproject.toml): finished with status 'done'
  Created wheel for prosperous_bot: filename=prosperous_bot-0.1.0-0.editable-py3-none-any.whl size=13362 sha256=312101976ac262637c04126ddaf2ad03a14066537c9622b475ee3911a79bd022
  Stored in directory: /tmp/pip-ephem-wheel-cache-kgya0b39/wheels/54/1b/b7/aa63e25c8f14f4f2ae7b04e6097bdecb770e455c5c1ee0a600
Successfully built prosperous_bot
Installing collected packages: prosperous_bot
Successfully installed prosperous_bot-0.1.0

[notice] A new release of pip is available: 25.1.1 -> 25.2
[notice] To update, run: pip install --upgrade pip
--- Environment Verification ---
+ echo '--- Environment Verification ---'
+ python --version
Python 3.12.11
+ pip --version
pip 25.1.1 from /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/pip (python 3.12)
+ pytest --version
pytest 8.4.2
--- Running initial test suite ---
+ echo '--- Running initial test suite ---'
+ pytest -m 'not integration' --cov
============================= test session starts ==============================
platform linux -- Python 3.12.11, pytest-8.4.2, pluggy-1.6.0
rootdir: /app
configfile: pytest.ini
testpaths: tests
plugins: asyncio-1.2.0, mock-3.15.1, anyio-4.10.0, cov-7.0.0, hypothesis-6.139.1
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 82 items / 2 deselected / 80 selected

tests/test_ci_probe.py ..                                                [  2%]
tests/test_exchange.py ............                                      [ 17%]
tests/test_exchange_api.py .........                                     [ 28%]
tests/test_exchange_property.py FFF                                      [ 32%]
tests/test_portfolio.py ...                                              [ 36%]
tests/test_portfolio_property.py .                                       [ 37%]
tests/test_rebalance.py .                                                [ 38%]
tests/test_rebalance_backtester_leverage.py ..........                   [ 51%]
tests/test_rebalance_engine.py ..                                        [ 53%]
tests/test_rebalance_signal_handling.py ...........                      [ 67%]
tests/test_simulate_rebalance_pnl.py .                                   [ 68%]
tests/test_utils.py .......................F.                            [100%]

=================================== FAILURES ===================================
______________________ test_create_futures_order_property ______________________

E   hypothesis.errors.FailedHealthCheck: 'tests/test_exchange_property.py::test_create_futures_order_property' uses a function-scoped fixture 'exch'.
    
    Function-scoped fixtures are not reset between inputs generated by `@given(...)`, which is often surprising and can cause subtle test bugs.
    
    If you were expecting the fixture to run separately for each generated input, then unfortunately you will need to find a different way to achieve your goal (for example, replacing the fixture with a similar context manager inside of the test).
    
    If you are confident that your test will work correctly even though the fixture is not reset between generated inputs, you can suppress this health check with @settings(suppress_health_check=[HealthCheck.function_scoped_fixture]). See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck for details.
All traceback entries are hidden. Pass `--full-trace` to see hidden and internal frames.
_______________________ test_create_spot_order_property ________________________

E   hypothesis.errors.FailedHealthCheck: 'tests/test_exchange_property.py::test_create_spot_order_property' uses a function-scoped fixture 'exch'.
    
    Function-scoped fixtures are not reset between inputs generated by `@given(...)`, which is often surprising and can cause subtle test bugs.
    
    If you were expecting the fixture to run separately for each generated input, then unfortunately you will need to find a different way to achieve your goal (for example, replacing the fixture with a similar context manager inside of the test).
    
    If you are confident that your test will work correctly even though the fixture is not reset between generated inputs, you can suppress this health check with @settings(suppress_health_check=[HealthCheck.function_scoped_fixture]). See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck for details.
All traceback entries are hidden. Pass `--full-trace` to see hidden and internal frames.
___________________________ test_positions_property ____________________________

E   hypothesis.errors.FailedHealthCheck: 'tests/test_exchange_property.py::test_positions_property' uses a function-scoped fixture 'exch'.
    
    Function-scoped fixtures are not reset between inputs generated by `@given(...)`, which is often surprising and can cause subtle test bugs.
    
    If you were expecting the fixture to run separately for each generated input, then unfortunately you will need to find a different way to achieve your goal (for example, replacing the fixture with a similar context manager inside of the test).
    
    If you are confident that your test will work correctly even though the fixture is not reset between generated inputs, you can suppress this health check with @settings(suppress_health_check=[HealthCheck.function_scoped_fixture]). See https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck for details.
All traceback entries are hidden. Pass `--full-trace` to see hidden and internal frames.
________________________ test_get_lot_step_api_success _________________________

mock_gate_client = <MagicMock name='gate_client' id='139970431240000'>

    @patch('prosperous_bot.exchange_gate.gate_client')
    def test_get_lot_step_api_success(mock_gate_client):
        mock_pair = MagicMock()
        mock_pair.min_base_amount = '0.001'
        mock_gate_client.get_spot_pairs.return_value = [mock_pair]
>       assert get_lot_step("BTC") == 0.001
E       AssertionError: assert 0.0001 == 0.001
E        +  where 0.0001 = get_lot_step('BTC')

tests/test_utils.py:70: AssertionError
=============================== warnings summary ===============================
tests/test_rebalance_backtester_leverage.py::test_compute_metrics_fallback_single_timestamp
  /home/jules/.pyenv/versions/3.12.11/lib/python3.12/site-packages/numpy/lib/_nanfunctions_impl.py:1214: RuntimeWarning:
  
  Mean of empty slice

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.11-final-0 _______________

Name                                          Stmts   Miss  Cover
-----------------------------------------------------------------
src/prosperous_bot/__init__.py                    0      0   100%
src/prosperous_bot/exchange_gate.py              89      7    92%
src/prosperous_bot/logging_config.py             13      0   100%
src/prosperous_bot/portfolio_manager.py          30      0   100%
src/prosperous_bot/rebalance_backtester.py      716    171    76%
src/prosperous_bot/rebalance_engine.py          128     27    79%
src/prosperous_bot/utils.py                      75     27    64%
tests/conftest.py                                14      0   100%
tests/test_ci_probe.py                            8      0   100%
tests/test_exchange.py                           74      0   100%
tests/test_exchange_api.py                       98      1    99%
tests/test_exchange_property.py                  85     63    26%
tests/test_neutral_rebalance.py                  18     11    39%
tests/test_portfolio.py                          96      4    96%
tests/test_portfolio_property.py                 17      0   100%
tests/test_rebalance.py                          24      1    96%
tests/test_rebalance_backtester_leverage.py     276     21    92%
tests/test_rebalance_engine.py                   53      4    92%
tests/test_rebalance_signal_handling.py         185     11    94%
tests/test_simulate_rebalance_pnl.py             32      0   100%
tests/test_utils.py                              39      1    97%
-----------------------------------------------------------------
TOTAL                                          2070    349    83%
=========================== short test summary info ============================
FAILED tests/test_exchange_property.py::test_create_futures_order_property - ...
FAILED tests/test_exchange_property.py::test_create_spot_order_property - hyp...
FAILED tests/test_exchange_property.py::test_positions_property - hypothesis....
FAILED tests/test_utils.py::test_get_lot_step_api_success - AssertionError: a...
============ 4 failed, 76 passed, 2 deselected, 1 warning in 9.24s =============
