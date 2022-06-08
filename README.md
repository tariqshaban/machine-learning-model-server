Embedding Trained Machine Learning Algorithms into Restful APIs
==============================
This is a submission of **assignment 1** for the **CIS726** course.

It contains the code necessary to host a primitive local restful API that utilizes machine learning models to predict
incoming request values.

Getting Started
------------
Clone the project from GitHub

`$ git clone https://github.com/tariqshaban/machine-learning-model-server.git`

It is encouraged to refer to [FastAPI](https://fastapi.tiangolo.com/tutorial/) documentation.

You may need to configure the Python interpreter (depending on the used IDE).

You may encounter problem concerning CORS policy when the server is improperly hosted.

No further configuration is required.

Usage
------------
Execute the `uvicorn main:app` command in the console, ensure that the port 8000 is not occupied, if need be, add
the `--port *YOUR_PORT*` flag.

You can also issue direct API request using [Heroku](https://www.heroku.com), for
example, [winning](https://machine-learning-model-server.herokuapp.com/machine_learning_models/winning?club1=Barcelona&club2=Real%20Madrid)
and
[goals](https://machine-learning-model-server.herokuapp.com/machine_learning_models/goals?shots=10&shots_on_target=20).

--------