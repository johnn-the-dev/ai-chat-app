'use client'

import { useState, useRef, useEffect } from 'react';

export default function ChatPage() {
    const [messages, setMessages] = useState<{ role: string, content: string }[]>([]);
    const [input, setInput] = useState('');
    const [userId, setUserId] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleLogin = (formData: FormData) => {
        const id = formData.get('userName') as string;
        if (id?.trim()) setUserId(id.trim());
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file || !userId) return;
        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('http://localhost:8000/upload/${userId}', {
                method: 'POST',
                body: formData,
            });
            
            if (response.ok) {
                alert("File successfully uploaded.");
            } else {
                alert("Error while uploading file.")
            }
        } catch (error) {
            console.error(error);
            alert("Backend not responding.")
        } finally {
            setUploading(false);
            e.target.value = '';
        }
    };

    const sendMessage = async () => {
        if (!input.trim() || !userId) return;
        const userMsg = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMsg]);
        const currentInput = input;
        setInput('');

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, message: currentInput }),
            });

            if (!response.ok) throw new Error();
            const data = await response.json();
            setMessages((prev) => [...prev, {role: 'ai', content: data.ai_response }]);
        } catch (error) {
            setMessages((prev) => [...prev, { role: 'ai', content: 'Error: Backend unavailable.'}]);
        }
    };

    if (!userId) {
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-[#212121] text-white p-4 font-sans">
                <form action={handleLogin} className="flex flex-col items-center w-full max-w-sm">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-8 shadow-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-[#212121]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h2 className="text-3xl font-semibold mb-2 text-center">Welcome back</h2>
                    <p className="text-gray-400 text-sm mb-8 text-center">Log in to continue your conversation</p>
                    
                    <input 
                        name="userName" 
                        className="w-full bg-transparent border border-gray-600 focus:border-white p-4 rounded-xl mb-4 text-white outline-none transition-all text-lg placeholder-gray-500"
                        placeholder="Enter user ID..."
                        required
                    />
                    <button type="submit" className="w-full bg-white hover:bg-gray-200 text-[#212121] py-4 rounded-xl font-bold transition-all text-lg shadow-md active:scale-[0.98]">
                        Continue
                    </button>
                </form>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-screen bg-[#212121] text-gray-100 font-sans selection:bg-gray-600">
            <div className="flex justify-between items-center p-4 sticky top-0 z-10 bg-[#212121]">
                <div className="flex items-center gap-2 group cursor-pointer">
                    <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                         <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#212121]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                    </div>
                    <h1 className="font-medium text-lg text-gray-200 group-hover:text-white transition-colors">AI Assistant <span className="text-xs text-gray-500 ml-2 bg-gray-800 px-2 py-1 rounded-md">v1.0</span></h1>
                </div>
                
                <div className="flex items-center gap-4">
                    <div className="text-sm font-medium text-gray-400 bg-[#2f2f2f] px-3 py-1.5 rounded-full">
                        {userId}
                    </div>
                    <button 
                        onClick={() => { setUserId(null); setMessages([]); }} 
                        className="p-2 hover:bg-[#2f2f2f] rounded-full text-gray-400 hover:text-white transition-colors"
                        title="Logout"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                        </svg>
                    </button>
                </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 w-full scrollbar-thin scrollbar-thumb-gray-600">
                <div className="max-w-3xl mx-auto space-y-6 pb-6">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                            <div className="w-16 h-16 bg-[#2f2f2f] rounded-full flex items-center justify-center mb-4">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                </svg>
                            </div>
                            <h3 className="text-xl font-medium text-gray-200 mb-2">How can I help you today?</h3>
                            <p className="text-gray-500">History loaded for {userId}. Upload a document or type a message.</p>
                        </div>
                    )}
                    
                    {messages.map((msg, i) => (
                        <div key={i} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-1`}>
                            {msg.role === 'ai' && (
                                <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center mr-4 flex-shrink-0 mt-1">
                                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#212121]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                    </svg>
                                </div>
                            )}

                            <div className={`max-w-[80%] px-5 py-3.5 text-[15px] leading-relaxed ${
                                msg.role === 'user' 
                                    ? 'bg-[#2f2f2f] text-gray-100 rounded-3xl rounded-tr-sm' 
                                    : 'bg-transparent text-gray-200'
                            }`}>
                                <p className="whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        </div>
                    ))}
                    <div ref={messagesEndRef} className="h-4" />
                </div>
            </div>

            <div className="p-4 bg-[#212121]">
                <div className="max-w-3xl mx-auto">
                    <div className="relative flex items-end bg-[#2f2f2f] rounded-[24px] border border-gray-700/50 shadow-sm focus-within:bg-[#383838] transition-colors p-2">
                        
                        <label className={`cursor-pointer p-3 rounded-full transition-colors flex items-center justify-center mb-0.5 ml-1 ${
                            uploading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-600 text-gray-400 hover:text-white'
                        }`} title="Upload PDF">
                            {uploading ? (
                                <svg className="animate-spin h-6 w-6 text-gray-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            ) : (
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 transform -rotate-45" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
                                </svg>
                            )}
                            <input type="file" className="hidden" onChange={handleFileUpload} accept=".pdf" disabled={uploading} />
                        </label>

                        {/* Textové pole */}
                        <textarea 
                            className="flex-1 max-h-48 bg-transparent border-none text-white focus:ring-0 resize-none py-3.5 px-2 outline-none placeholder-gray-500 overflow-y-auto min-h-[52px]"
                            value={input} 
                            onChange={(e) => setInput(e.target.value)} 
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    sendMessage();
                                }
                            }}
                            placeholder="Message AI Agent..."
                            rows={1}
                        />

                        <button 
                            onClick={sendMessage} 
                            disabled={!input.trim()}
                            className={`p-2 rounded-full mb-1.5 mr-1.5 transition-all flex items-center justify-center ${
                                input.trim() ? 'bg-white text-black hover:bg-gray-200' : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                            }`}
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 10l7-7m0 0l7 7m-7-7v18" />
                            </svg>
                        </button>
                    </div>
                    <div className="text-center mt-3">
                        <span className="text-xs text-gray-500">AI Agent can make mistakes. Consider verifying important information.</span>
                    </div>
                </div>
            </div>
        </div>
    );
}