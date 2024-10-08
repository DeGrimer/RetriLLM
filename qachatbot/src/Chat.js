import React, { useState } from 'react';

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");

    const sendMessage = async () => {
        const response = await fetch("http://localhost:80/chatbot/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: input })
        });
        const data = await response.json();
        setMessages([...messages, { text: input, from: "user" }, { text: data.response, from: "bot" }]);
        setInput("");
    };

    return (
        <div>
            <div>
                {messages.map((msg, index) => (
                    <div key={index} className={msg.from === "user" ? "user-message" : "bot-message"}>
                        {msg.text.split('\n').map((item, idx) => { return (
                        <React.Fragment key={idx}>
                            {item}
                            <br />
                        </React.Fragment>
                        )
                        })}
                    </div>
                ))}
            </div>
            <input 
                value={input} 
                onChange={(e) => setInput(e.target.value)} 
                onKeyUp={(e) => e.key === 'Enter' && sendMessage()} 
            />
            <button onClick={sendMessage}>Send</button>
        </div>
    );
};

export default Chatbot;