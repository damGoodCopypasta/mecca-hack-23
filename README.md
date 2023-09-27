# mecca-hack-23


### Server
To start the server.
1. open the server subfolder in vscode
2. python3 -m venv venv to create a virtual env
    ```bash
    python3 -m venv venv
    ```
3. execute /venv/bin/activate to enter env
    ```bash
    source /venv/bin/activate
    ```
4. 
    ```bash
    pip install -r requirements.txt
    ```
5. 
    ```bash
    litestar run
    ```

You should be able to hit /message with a {query: ""} payload to get the bot to respond.


### Teams Chat Bot
To start the chat bot

1. open the teams subfolder in vscode
2. Prerequisites: to install [Bot Framework Emulator](https://github.com/Microsoft/BotFramework-Emulator/releases/tag/v4.14.1) for testing purpose
3. Follow the README file under subfolder