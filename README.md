# Project Overview
GitPilot is an AI Agent that automates the task of performing GitHub workflows by user prompt and the user no longer needs to remember the hard syntax of GitHub CLI commands. 
Final Project Docker Repository Link : https://hub.docker.com/repository/docker/adisrinitw/gitpilot30-gitpilot/general
Porject GitHub Repository Link: https://github.com/a-code-sri/GitPilot
# Team Members and Contributions
1. 24CSB1A04 Aditya Shrivastava  
    **File Contributions** :  
      1. Backend  : Complete  
      2. LLM_training : api.py
      3. Dockerfile , docker-compose.yml , app.py , agent_context.json
2. 24CSB1A42 Mummasani Avinash Reddy
    **File Contributions**  :
       1. Frontend : Complete  
       2. LLM_training : *training.py* , *dpo_training_data.json*
       3. Research_paper.pdf
# Tech Stack  
**Frontend**          : ReactJS  
**Backend**           : Flask  
**HuggingFace Model** : [Qwen 7b](https://huggingface.co/Qwen/Qwen2-7B-Instruct)   
**Voice assistance**  : WebAPI  
**Deployment**        : Docker
# Project Architecture   
**Frontend**  : The frontend made using ReactJs framework takes the user input either through prompt or through voice assistance.  
                In case of voice input, WebAPI converts the voice input into text input and finally the frontend sends its request to the Backend.  
**LLM_Training** : 
     The trained LLM model is hosted on a seperate API   
     It consists of the following files : 
        1. *api.py*  : The main file for LLM inference and hosting
        2. *training.py* : Loading training data and training the LLM Model.  
        3. *dpo_training.json* : Original Training data
**Backend**   : The backend makes a POST request to another API that hosts our trained LLM Model. The API responds with the set of actions to be taken and the corresponding parameters. The backend then processes these actions and executes GitHub CLI commands on it's own.  
       It's strucutre is given by :  
          */backend*
             */agent* 
                  */planner.py* : Responsible for making request to the LLM Model based on user request and returns LLM json
                  repsonse consisting of the set of actions.  
             */github*   
                   */github_api.py* : Converts a specified action(eg: create_repo) to GitHub CLI Command   
                   */github_cli.py* : Runs the GitHub CLI Command in user system and commits it to the original repository  
**Files in Project main directory** :  
        *app.py*   : The main file to host the backend Flask App to receive frontend request  
        *agent_context.json* : A file that helps agent remember context to extract required parameters to perform an action
                                (eg. Agent might need repo_name parameter while performing action delete_repo)
        *DockerFile*  : It is the Docker image of the project. Used for project deployment  
        *docker-compose.yml* : It is the environment file containing all dependencies.   
        *Research_paper.pdf* : The source CVPR research paper.

# Requirements
1. GPU with minimum **15 GB VRAM for training the LLM model** and hosting it on API.
2. **Docker Desktop** if you want to access the final deployed version using Docker.
# How to train llm and setup llm server?
1. Open *training.py* file and change paths according to current working directory. 
Note: change model_path="Qwen/Qwen2-7B" is for the first time use and change it to it stored path for later use.
2. Run *training.py* for completing training it takes approximately 2 to 3 hours.
3. Run *api.py* to host the trained LLM model on a seperate API.
4. Paste the url shown by api.py to the location *backend/agent/planner.py* to the variable *EXTERNAL_API_URL*(not required if using docker).
(since we used free server, the url is not static and hence keep checking it and update the url in backend file accordingly).

# How to use the project?(after hosting the LLM API)   
First make sure you have downloaded the given zip file. Unzip it and open project root directory.
**Method-1 Using Docker: (no other dependencies needed other than Docker Desktop)**
1. Run the following commands from project **root directory**: 
   *docker pull adisrinitw/gitpilot30-gitpilot:latest*
   *docker run -it -e EXTERNAL_API_URL=<YOUR_API_URL> -p 5173:5173 -p 5000:5000 adisrinitw/gitpilot30-gitpilot:latest*
2. Click on the link(http://localhost:5173/) to open project frontend. You can now use the agent after authenticating your GitHub.
**Method-2 Manual Method(Requires manual installation of dependencies)**
1. Ensure all the below dependencies are already installed in your system: 
    a. **NodeJS**
    b. **ReactJS**
    c. **TailwindCSS**
    d. **Flask**
    e. **GitHub CLI and gh**
2. Run the following commands to install remaining dependcies after opening project **root directory** : 
    *cd backend*
    *pip install -r requirements.txt*
3. To host the backend run the following command from project **root directory**: 
   *python app.py*
4. To host the frontend run the following command from project **root directory** in a **seperate terminal**:
   *cd frontend*
   *npm run dev*
5. Click on the link(http://localhost:5173/) to open project frontend. You can now use the agent after authenticating your GitHub.
# References  
The architecture for the LLM Model has been adapted from [Section 2.2 Part-1 of this CVPR Research Paper](https://github.com/a-code-sri/GitPilot/blob/7c53ea88de8a676971acf650ea39f20d03c1343e/Research_paper.pdf)
