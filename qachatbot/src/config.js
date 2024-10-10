import { createChatBotMessage } from 'react-chatbot-kit';
const config = {  
    initialMessages: [createChatBotMessage(`Здесь вы можете задать вопрос, связанный с Трудовым Кодексом`)],
    customComponents: {     
        // Replaces the default header    
        header: () => <div style={{ backgroundColor: 'lightgrey', padding: "5px", borderRadius: "3px" }}>Консультация по ТК РФ</div>
    }
};
export default config;