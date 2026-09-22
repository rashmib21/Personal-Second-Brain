import React, { useEffect, useMemo, useState } from "react";
import {
  Brain, ChevronLeft, ChevronRight, FileText, Files, Image as ImageIcon,
  MessageSquare, PanelLeft, Search, Send, Sheet, Sparkles, X
} from "lucide-react";
import {
  askRag, getDocuments, getFiles, getImages, getSpreadsheets
} from "./api";

const views = [
  { id: "global", label: "All Chats", icon: MessageSquare },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "spreadsheets", label: "Spreadsheets", icon: Sheet },
  { id: "images", label: "Images", icon: ImageIcon }
];

function normalizeList(payload, keys = []) {
  if (Array.isArray(payload)) return payload;
  for (const key of keys) {
    if (Array.isArray(payload?.[key])) return payload[key];
  }
  return [];
}

function itemName(item) {
  if (typeof item === "string") return item;
  return item?.name || item?.filename || item?.file_name || item?.source || item?.path || "Untitled";
}

function itemPath(item) {
  if (typeof item === "string") return item;
  return item?.path || item?.canonical_path || item?.source || item?.filename || item?.name || "";
}

export default function App() {
  const [view, setView] = useState("global");
  const [selectedSource, setSelectedSource] = useState(null);
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [items, setItems] = useState([]);
  const [inventoryError, setInventoryError] = useState("");
  const [loadingInventory, setLoadingInventory] = useState(false);
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    let alive = true;

    async function load() {
      setLoadingInventory(true);
      setInventoryError("");

      try {
        let data;
        if (view === "documents") data = await getDocuments();
        else if (view === "spreadsheets") data = await getSpreadsheets();
        else if (view === "images") data = await getImages();
        else data = await getFiles();

        const keys =
          view === "documents" ? ["documents", "files", "items"] :
          view === "spreadsheets" ? ["spreadsheets", "files", "items"] :
          view === "images" ? ["images", "files", "items"] :
          ["files", "items"];

        if (alive) setItems(normalizeList(data, keys));
      } catch (error) {
        if (alive) {
          setItems([]);
          setInventoryError(error.message);
        }
      } finally {
        if (alive) setLoadingInventory(false);
      }
    }

    load();
    return () => { alive = false; };
  }, [view]);

  const title = useMemo(() => {
    if (selectedSource) return selectedSource.name;
    return views.find((x) => x.id === view)?.label || "All Chats";
  }, [view, selectedSource]);

  function selectView(next) {
    setView(next);
    setSelectedSource(null);
    setMessages([]);
  }

  function selectSource(item) {
    setSelectedSource({
      name: itemName(item),
      path: itemPath(item),
      type: view
    });
    setMessages([]);
  }

  async function sendMessage(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text || sending) return;

    setMessages((current) => [...current, { role: "user", text }]);
    setQuery("");
    setSending(true);

    try {
      // Determine active_file when a sidebar source is selected
      let activeFile = null;
      if (selectedSource) {
        activeFile = selectedSource.path || selectedSource.name || null;
      }

      // Determine modality context (selected source type, or active view if document/spreadsheet/image)
      let modalityValue = null;
      if (selectedSource) {
        modalityValue = selectedSource.type || null;
      } else if (view === "document" || view === "spreadsheet" || view === "image") {
        modalityValue = view;
      }

      const response = await askRag(text, {
        active_file: activeFile,
        modality: modalityValue
      });

      const answer =
        response?.answer ??
        response?.response ??
        response?.message ??
        (typeof response === "string" ? response : "The backend returned no answer.");

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          text: answer,
          sources: response?.sources || response?.source || []
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: "error", text: error.message }
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="brand">
          <div className="brand-mark"><Brain size={19} /></div>
          {sidebarOpen && (
            <div>
              <div className="brand-title">Second Brain</div>
              <div className="brand-subtitle">Local RAG workspace</div>
            </div>
          )}
        </div>

        <nav className="view-nav">
          {views.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${view === id ? "active" : ""}`}
              onClick={() => selectView(id)}
              title={label}
            >
              <Icon size={18} />
              {sidebarOpen && <span>{label}</span>}
            </button>
          ))}
        </nav>

        {sidebarOpen && (
          <div className="sidebar-section">
            <div className="section-heading">
              <span>{selectedSource ? "Focused source" : "Available sources"}</span>
              <Files size={14} />
            </div>

            {selectedSource ? (
              <button className="source-card selected" onClick={() => setSelectedSource(null)}>
                <FileText size={16} />
                <span title={selectedSource.name}>{selectedSource.name}</span>
                <X size={14} />
              </button>
            ) : (
              <div className="source-list">
                {loadingInventory && <div className="muted">Loading sources…</div>}
                {!loadingInventory && inventoryError && (
                  <div className="inventory-error">
                    {inventoryError}
                    <small>Configure the inventory routes in <code>src/api.js</code>.</small>
                  </div>
                )}
                {!loadingInventory && !inventoryError && items.length === 0 && (
                  <div className="muted">No sources returned by the backend.</div>
                )}
                {!loadingInventory && items.map((item, index) => (
                  <button
                    key={`${itemName(item)}-${index}`}
                    className="source-card"
                    onClick={() => selectSource(item)}
                    title={itemName(item)}
                  >
                    {view === "images" ? <ImageIcon size={16} /> :
                     view === "spreadsheets" ? <Sheet size={16} /> :
                     <FileText size={16} />}
                    <span>{itemName(item)}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <button className="collapse-button" onClick={() => setSidebarOpen((v) => !v)}>
          {sidebarOpen ? <ChevronLeft size={17} /> : <ChevronRight size={17} />}
        </button>
      </aside>

      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-toggle" onClick={() => setSidebarOpen((v) => !v)}>
              <PanelLeft size={18} />
            </button>
            <div>
              <div className="eyebrow">WORKSPACE</div>
              <h1>{title}</h1>
            </div>
          </div>

          <div className="connection-pill">
            <span className="status-dot" />
            Local RAG
          </div>
        </header>

        <section className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon"><Sparkles size={23} /></div>
              <div className="welcome-kicker">SECOND BRAIN</div>
              <h2>
                {selectedSource
                  ? `Ask anything about ${selectedSource.name}`
                  : view === "images"
                    ? "Search your images naturally."
                    : `Search your ${views.find((x) => x.id === view)?.label.toLowerCase() || "knowledge"} naturally.`}
              </h2>
              <p>
                {selectedSource
                  ? "This chat stays scoped to the selected source."
                  : "Retrieval, source resolution, grounding, and answers stay inside your existing RAG backend."}
              </p>

              <div className="suggestions">
                {view === "images" && !selectedSource && (
                  <>
                    <button onClick={() => setQuery("What’s in the mummy photo?")}>What’s in the mummy photo?</button>
                    <button onClick={() => setQuery("Show me the image with Krishna.")}>Show me the image with Krishna.</button>
                  </>
                )}
                {view === "spreadsheets" && !selectedSource && (
                  <>
                    <button onClick={() => setQuery("Top 5 companies by CTC")}>Top 5 companies by CTC</button>
                    <button onClick={() => setQuery("List all spreadsheets")}>List all spreadsheets</button>
                  </>
                )}
                {view === "documents" && !selectedSource && (
                  <>
                    <button onClick={() => setQuery("Where did I mention joins?")}>Where did I mention joins?</button>
                    <button onClick={() => setQuery("What did I write about backend development?")}>What did I write about backend development?</button>
                  </>
                )}
                {view === "global" && (
                  <>
                    <button onClick={() => setQuery("Where did I mention Kafka?")}>Where did I mention Kafka?</button>
                    <button onClick={() => setQuery("How many files do I have?")}>How many files do I have?</button>
                  </>
                )}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((message, index) => (
                <div key={index} className={`message-row ${message.role}`}>
                  <div className="message-avatar">
                    {message.role === "user" ? "R" :
                     message.role === "error" ? "!" : <Brain size={16} />}
                  </div>
                  <div className="message-content">
                    <div className="message-label">
                      {message.role === "user" ? "You" :
                       message.role === "error" ? "Backend error" : "Second Brain"}
                    </div>
                    <div className="message-text">{message.text}</div>

                    {message.sources?.length > 0 && (
                      <div className="sources">
                        <span>Sources</span>
                        {Array.isArray(message.sources)
                          ? message.sources.map((source, sourceIndex) => (
                              <span className="source-chip" key={sourceIndex}>
                                {typeof source === "string"
                                  ? source
                                  : source?.source || source?.name || "Source"}
                              </span>
                            ))
                          : <span className="source-chip">{String(message.sources)}</span>}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {sending && (
                <div className="message-row assistant">
                  <div className="message-avatar"><Brain size={16} /></div>
                  <div className="typing"><span /><span /><span /></div>
                </div>
              )}
            </div>
          )}
        </section>

        <form className="composer" onSubmit={sendMessage}>
          {selectedSource && (
            <div className="scope-pill">
              <FileText size={14} />
              Scoped to <strong>{selectedSource.name}</strong>
              <button type="button" onClick={() => setSelectedSource(null)}><X size={13} /></button>
            </div>
          )}

          <div className="composer-box">
            <Search size={18} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={selectedSource ? "Ask a question about this source…" : "Ask your Second Brain…"}
              disabled={sending}
            />
            <button className="send-button" type="submit" disabled={!query.trim() || sending}>
              <Send size={17} />
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}