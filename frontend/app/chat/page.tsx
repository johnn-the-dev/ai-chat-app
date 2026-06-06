'use client'

import { useState, useRef, useEffect } from 'react';

interface ChatSession {
    id: string;
    title: string;
}

export default function ChatPage() {
    const [messages, setMessages] = useState<{ role: string, content: string }[]>([]);
    const [input, setInput] = useState('');
    const [userId, setUserId] = useState<string | null>(null);
    const [uploading, setUploading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [isRegistering, setIsRegistering] = useState(false);
    const [authError, setAuthError] = useState('');

    const [documents, setDocuments] = useState<string[]>([]);
    const [showDocs, setShowDocs] = useState(false);

    const [editingChatId, setEditingChatId] = useState<string | null>(null);
    const [editTitle, setEditTitle] = useState('');

    const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
    const [activeThreadId, setActiveThreadId] = useState<string | null>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        if (showDocs && userId) {
            fetchDocuments();
        }
    }, [showDocs, userId]);

    useEffect(() => {
        if (activeThreadId && userId) {
            const token = localStorage.getItem('token');
            if (token) {
                fetchHistory(activeThreadId, token);
            }
        } else {
            setMessages([]);
        }
    }, [activeThreadId, userId]);

    const handleLogout = () => {
        localStorage.removeItem('token');
        setUserId(null);
        setMessages([]);
        setUsername('');
        setPassword('');
        setChatSessions([]);
        setActiveThreadId(null);
    };

    const fetchChatSessions = async (token: string) => {
        try {
            const response = await fetch(`http://localhost:8000/chats`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setChatSessions(data);

                if (data.length > 0) {
                    setActiveThreadId(data[0].id);
                } else {
                    createNewChat(token);
                }
            }
        } catch (error) {
        console.error("Error fetching chat sessions:", error);
        } 
    };

    const renameChat = async (threadId: string, newTitle: string) => {
        if (!newTitle.trim()) {
            setEditingChatId(null);
            return;
        }
        try {
            const token = localStorage.getItem('token');
            const response = await fetch(`http://localhost:8000/chats/${threadId}`, {
                method: 'PUT',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}` 
                },
                body: JSON.stringify({ title: newTitle })
            });

            if (response.ok) {
                setChatSessions(prev => prev.map(chat => 
                    chat.id === threadId ? { ...chat, title: newTitle } : chat
                ));
            }
        } catch (error) {
            console.error("Error renaming chat:", error);
        }
        setEditingChatId(null);
    };

    const createNewChat = async (tokenParam?: string) => {
        const token = tokenParam || localStorage.getItem('token');
        if (!token) return;

        try {
            const response = await fetch(`http://localhost:8000/chats`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const newChat = await response.json();
                setChatSessions(prev => [newChat, ...prev]);
                setActiveThreadId(newChat.id);
                setMessages([]);
            }
        } catch (error) {
            console.error("Error creating new chat:", error);
        }
    };

    const fetchHistory = async (threadId: string, token: string) => {
        try {
            const response = await fetch(`http://localhost:8000/history/${threadId}`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });
            if (response.ok) {
                const data = await response.json();
                const formattedMessages: {role: string, content: string}[] = [];

                data.forEach((item: any) => {
                    formattedMessages.push({ role: 'user', content: item.user_message });
                    formattedMessages.push({ role: 'ai', content: item.ai_response });
                });

                setMessages(formattedMessages);
            } else if (response.status === 404) {
                setMessages([]);
            } else {
                console.error("Failed to load chat history.");
            }
        } catch (error) {
            console.error("Error: backend error (chat history)")
        }
    };

    const fetchDocuments = async () => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch (`http://localhost:8000/documents/${userId}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setDocuments(data.documents || []);
            }
        } catch (error) {
            console.error("Error loading documents:", error);
        }
    };

    const deleteDocument = async (filename: string) => {
        try {
            const token = localStorage.getItem('token');
            const response = await fetch (`http://localhost:8000/documents/${userId}/${filename}`, {
                method: 'DELETE',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                setDocuments((prev) => prev.filter(doc => doc !== filename));
            } else {
                alert("Failed to load document.");
            }
        } catch (error) {
            console.error("Error deleting document:", error);
        }
    };

    const handleAuth = async (e: React.SyntheticEvent<HTMLFormElement>) => {
        e.preventDefault();
        setAuthError('');

        try {
            if (isRegistering) {
                const res = await fetch('http://localhost:8000/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                if (!res.ok) {
                    const data = await res.json();
                    throw new Error(data.detail || 'Registration failed');
                }
                
                alert('Successfully registered! You can now log in.');
                setIsRegistering(false);
                setPassword('');
                
            } else {
                const formData = new URLSearchParams();
                formData.append('username', username);
                formData.append('password', password);

                const res = await fetch('http://localhost:8000/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                    body: formData
                });
                
                if (!res.ok) {
                    const data = await res.json();
                    throw new Error(data.detail || 'Login failed');
                }
                
                const data = await res.json();
                localStorage.setItem('token', data.access_token);
                setUserId(data.username);
                await fetchChatSessions(data.access_token);
            }
        } catch (err: any) {
            setAuthError(err.message);
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file || !userId) return;
        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);
        const token = localStorage.getItem('token');

        try {
            const response = await fetch(`http://localhost:8000/upload/${userId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData,
            });
            
            if (response.ok) {
                alert("File successfully uploaded.");
            } else if (response.status === 401){
                alert("Session expired, please login again.");
                handleLogout();
            } else {
                alert("Error while uploading file.");
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
        const token = localStorage.getItem('token')

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}` 

                },
                body: JSON.stringify({ thread_id: activeThreadId, message: currentInput }),
            });

            if (!response.ok) {
                if (response.status === 401) {
                    alert("Session expired. Please log in again.");
                    handleLogout();
                    throw new Error('Unauthorized');
                }
                throw new Error('Failed to send message');
            }
            const data = await response.json();
            setMessages((prev) => [...prev, {role: 'ai', content: data.ai_response }]);

            if (data.chat_title) {
                setChatSessions(prev =>
                    prev.map(chat => chat.id === activeThreadId ? { ... chat, title: data.chat_title }: chat)
                );
            }
        } catch (error) {
            setMessages((prev) => [...prev, { role: 'ai', content: 'Error: Backend unavailable or unauthorized.'}]);
        }
    };

    if (!userId) {
        return (
            <div className="flex flex-col items-center justify-center h-screen bg-[#212121] text-white p-4 font-sans">
                <form onSubmit={handleAuth} className="flex flex-col items-center w-full max-w-sm">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-8 shadow-lg">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-[#212121]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                        </svg>
                    </div>
                    <h2 className="text-3xl font-semibold mb-2 text-center">
                        {isRegistering ? 'Create Account' : 'Welcome back'}
                    </h2>
                    <p className="text-gray-400 text-sm mb-6 text-center">
                        {isRegistering ? 'Register to start chatting' : 'Log in to continue your conversation'}
                    </p>
                    
                    {authError && (
                        <div className="w-full bg-red-900/50 border border-red-500 text-red-200 p-3 rounded-xl mb-4 text-sm text-center">
                            {authError}
                        </div>
                    )}

                    <input 
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        className="w-full bg-[#2f2f2f] border border-gray-600 focus:border-white p-4 rounded-xl mb-4 text-white outline-none transition-all text-lg placeholder-gray-500"
                        placeholder="Username"
                        required
                    />
                    <input 
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full bg-[#2f2f2f] border border-gray-600 focus:border-white p-4 rounded-xl mb-6 text-white outline-none transition-all text-lg placeholder-gray-500"
                        placeholder="Password"
                        required
                    />
                    <button type="submit" className="w-full bg-white hover:bg-gray-200 text-[#212121] py-4 rounded-xl font-bold transition-all text-lg shadow-md active:scale-[0.98]">
                        {isRegistering ? 'Register' : 'Login'}
                    </button>

                    <p className="mt-6 text-gray-400 text-sm cursor-pointer hover:text-white transition-colors" onClick={() => { setIsRegistering(!isRegistering); setAuthError(''); }}>
                        {isRegistering ? 'Already have an account? Log in' : "Don't have an account? Register"}
                    </p>
                </form>
            </div>
        );
    }

    return (
        <div className="flex h-screen bg-[#212121] text-gray-100 font-sans selection:bg-gray-600">
            
            <div className="w-64 bg-[#171717] border-r border-gray-800 flex flex-col hidden md:flex">
                <div className="p-4">
                     <button 
                        onClick={() => createNewChat()}
                        className="w-full bg-transparent border border-gray-700 hover:bg-[#2f2f2f] text-white py-2.5 rounded-xl font-medium transition-colors flex items-center justify-center gap-2"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        New Chat
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1 scrollbar-thin scrollbar-thumb-gray-700">
                    <p className="text-xs font-semibold text-gray-500 px-2 mb-2 uppercase tracking-wider">Your Chats</p>
                    {chatSessions.map((chat) => (
                        <div key={chat.id} className={`group relative flex items-center justify-between w-full px-3 py-2.5 rounded-lg text-sm transition-colors ${
                                activeThreadId === chat.id ? 'bg-[#2f2f2f] text-white' : 'text-gray-400 hover:bg-[#2f2f2f]/50 hover:text-gray-200'
                            }`}>
                            {editingChatId === chat.id ? (
                                <input
                                    type="text"
                                    value={editTitle}
                                    onChange={(e) => setEditTitle(e.target.value)}
                                    onBlur={() => renameChat(chat.id, editTitle)}
                                    onKeyDown={(e) => {
                                        if (e.key === 'Enter') renameChat(chat.id, editTitle);
                                        if (e.key === 'Escape') setEditingChatId(null);
                                    }}
                                    className="flex-1 bg-transparent border-none text-white outline-none focus:ring-0 px-0 h-full"
                                    autoFocus
                                />
                            ) : (
                                <>
                                    <button onClick={() => setActiveThreadId(chat.id)} className="flex-1 text-left truncate pr-2">
                                        {chat.title}
                                    </button>
                                    
                                    {(activeThreadId === chat.id) && (
                                        <button 
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                setEditingChatId(chat.id);
                                                setEditTitle(chat.title);
                                            }}
                                            className="text-gray-400 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                                            title="Rename chat"
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                                            </svg>
                                        </button>
                                    )}
                                </>
                            )}
                        </div>
                    ))}
                </div>

                <div className="p-4 border-t border-gray-800">
                     <div className="flex items-center gap-3 px-2">
                        <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center text-white font-bold text-sm">
                            {userId.charAt(0).toUpperCase()}
                        </div>
                        <div className="flex-1 truncate">
                             <p className="text-sm font-medium text-white truncate">{userId}</p>
                        </div>
                     </div>
                </div>
            </div>

            <div className="flex-1 flex flex-col relative min-w-0">
                <div className="flex justify-between items-center p-4 sticky top-0 z-10 bg-[#212121] border-b border-gray-800">
                    <div className="flex items-center gap-2 group cursor-pointer md:hidden">
                        <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-[#212121]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                            </svg>
                        </div>
                        <h1 className="font-medium text-lg text-gray-200 group-hover:text-white transition-colors">AI Assistant</h1>
                    </div>
                    <div className="hidden md:block">
                    </div>
                    
                    <div className="flex items-center gap-3">
                        <button 
                            onClick={() => setShowDocs(!showDocs)}
                            className={`p-2 rounded-full transition-colors flex items-center gap-2 text-sm font-medium ${showDocs ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-[#2f2f2f] text-gray-400 hover:text-white'}`}
                            title="Your Documents"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                        </button>

                        <button 
                            onClick={handleLogout}
                            className="p-2 hover:bg-[#2f2f2f] rounded-full text-gray-400 hover:text-white transition-colors"
                            title="Logout"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                        </button>
                    </div>
                </div>

                {showDocs && (
                    <div className="absolute right-4 top-20 w-80 bg-[#2f2f2f] border border-gray-700 shadow-2xl rounded-2xl z-20 p-5 animate-in fade-in slide-in-from-top-4">
                        <div className="flex justify-between items-center mb-4 border-b border-gray-700 pb-2">
                            <h3 className="text-white font-medium flex items-center gap-2">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                                </svg>
                                Vector Database
                            </h3>
                            <button onClick={() => setShowDocs(false)} className="text-gray-400 hover:text-white">✕</button>
                        </div>
                        
                        {documents.length === 0 ? (
                            <p className="text-gray-500 text-sm text-center py-4">No documents uploaded yet.</p>
                        ) : (
                            <ul className="space-y-2 max-h-64 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 pr-2">
                                {documents.map((doc, idx) => (
                                    <li key={idx} className="flex justify-between items-center bg-[#212121] p-3 rounded-xl border border-gray-700 group">
                                        <span className="text-gray-300 text-sm truncate pr-2" title={doc}>{doc}</span>
                                        <button 
                                            onClick={() => deleteDocument(doc)}
                                            className="text-gray-500 hover:text-red-400 transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                                            title="Delete file"
                                        >
                                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                            </svg>
                                        </button>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                )}
                
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
                                <p className="text-gray-500">History loaded. Upload a document or type a message.</p>
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
        </div>
    );
}