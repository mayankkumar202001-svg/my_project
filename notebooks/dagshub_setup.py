import dagshub
import mlflow
dagshub.init(repo_owner='mayankkumar202001-svg', repo_name='my_project', mlflow=True)

import mlflow
with mlflow.start_run(run_name="check run, not for main experiment "):
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
print("dagshub_setup.py started")  