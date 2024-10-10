import React from 'react';
const ActionProvider = ({ createChatBotMessage, setState, children }) => {  
    const sendMessage = async (input) => {
        const response = await fetch("http://localhost:80/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: input })
        });
        const data = await response.json();
        const botMessage = createChatBotMessage(data.response)
        setState((prev) => ({      
            ...prev,      
            messages: [...prev.messages, botMessage],    
        }));
    };
    return (    
    <div>      
        {React.Children.map(children, (child) => {        
            return React.cloneElement(child, {          
                actions: {
                    sendMessage
                },        
            });      
        })}    
        </div>  
        );
    };
export default ActionProvider;