// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
const { ActivityHandler, MessageFactory } = require('botbuilder');

class EchoBot extends ActivityHandler {
    constructor() {
        super();
        // See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
        this.onMessage(async (context, next) => {
            // request payload from incoming message
            const payload = { query: context.activity.text };

            // TODO: Replace this with the actual server endpoint url (maybe via ngrok)
            // server endpoint URL
            const URL = 'http://127.0.0.1:8000/message';

            // request options
            const options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            };

            // dynamic import node-fetch
            const nodeFetch = await import('node-fetch');
            const fetch = nodeFetch.default; // access the default export

            // fetch chat server endpoint
            const res = await fetch(URL, options).then();
            if (!res.ok) {
                throw new Error('Something went wrong!');
            }
            const responseObject = await res.json();

            // validate response object
            if (responseObject.links && responseObject.links.length > 0) {
                // generating links text
                let moreResources = `\nFor more information, you can check the links below:
                `;

                responseObject.links.forEach((item, index) => {
                    if (index > 0) {
                        moreResources += `\n${ item }`;
                    }
                });

                // combining response message and links
                const text = `*${ responseObject.message }*
                \nHere is a [link](${ responseObject.links[0] }) most relevant to your question
                ${ responseObject.links.length > 1 && moreResources }`;

                await context.sendActivity(MessageFactory.text(text, text));
            } else {
                await context.sendActivity(MessageFactory.text('Something went wrong', 'Something went wrong'));
            }

            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });

        this.onMembersAdded(async (context, next) => {
            const membersAdded = context.activity.membersAdded;
            const welcomeText = `Hello and welcome!
            \n What can I help for you?`;
            for (let cnt = 0; cnt < membersAdded.length; ++cnt) {
                if (membersAdded[cnt].id !== context.activity.recipient.id) {
                    await context.sendActivity(MessageFactory.text(welcomeText, welcomeText));
                }
            }
            // By calling next() you ensure that the next BotHandler is run.
            await next();
        });
    }
}

module.exports.EchoBot = EchoBot;
