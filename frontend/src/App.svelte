<script lang="ts">
  import { onDestroy } from "svelte";

  type UploadStatus =
    | "uploading"
    | "queued"
    | "processing"
    | "completed"
    | "failed";

  type Upload = {
    id: string;
    file: File;
    status: UploadStatus;
    documentId?: string;
    chunksIndexed?: number;
    error?: string;
  };

  type GroundingStatus = "supported" | "potential_drift" | "unverified";

  type ChatMessage = {
    id: string;
    role: "user" | "assistant";
    text: string;
    sources?: string[];
    groundingStatus?: GroundingStatus;
    loading?: boolean;
    error?: boolean;
  };

  const allowedExtensions = new Set(["pdf", "pptx", "docx", "txt", "md"]);
  let maxUploadBytes = $state(50_000_000);
  const threadId = globalThis.crypto?.randomUUID?.() ?? `web-${Date.now()}`;
  const suggestions = [
    "Give me a short summary of the key points.",
    "What are the most important dates or deadlines?",
    "What should I know before getting started?",
  ];

  let uploads = $state<Upload[]>([]);
  let messages = $state<ChatMessage[]>([]);
  let ingestKey = $state("local-ingest-key");
  let chatKey = $state("");
  let uploadFeedback = $state("");
  let connection = $state<"checking" | "online" | "offline">("checking");
  let message = $state("");
  let isSending = $state(false);
  let isDragging = $state(false);
  let toast = $state("");
  let toastError = $state(false);
  let messageInput: HTMLTextAreaElement;
  let messagesElement: HTMLElement;
  let toastTimer: ReturnType<typeof setTimeout>;
  const activeUploadControllers = new Set<AbortController>();
  const connectionTimer = setInterval(() => void checkConnection(), 30000);

  let documentCount = $derived(String(uploads.length).padStart(2, "0"));

  onDestroy(() => {
    clearInterval(connectionTimer);
    for (const controller of activeUploadControllers) {
      controller.abort();
    }
    activeUploadControllers.clear();
  });

  function errorMessage(error: unknown, fallback: string): string {
    return error instanceof Error ? error.message : fallback;
  }

  function showToast(text: string, isError = false): void {
    clearTimeout(toastTimer);
    toast = text;
    toastError = isError;
    toastTimer = setTimeout(() => {
      toast = "";
    }, 4200);
  }

  async function responseError(response: Response): Promise<string> {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as {
        detail?: unknown;
        error?: unknown;
      };
      const rawDetail = body?.detail ?? body?.error;
      if (typeof rawDetail === "string" && rawDetail) {
        detail = rawDetail;
      } else if (Array.isArray(rawDetail)) {
        const messages = rawDetail
          .map(
            (item) =>
              typeof item === "object" && item !== null && "msg" in item
                ? String((item as { msg?: unknown }).msg)
                : String(item),
          )
          .filter(Boolean);
        if (messages.length) {
          detail = messages.join("; ");
        }
      }
    } catch {
      return detail;
    }
    return detail;
  }

  function ingestionHeaders(): Record<string, string> {
    const key = ingestKey.trim();
    return key ? { "X-Ingest-API-Key": key } : {};
  }

  function updateUpload(id: string, update: Partial<Upload>): void {
    uploads = uploads.map((upload) =>
      upload.id === id ? { ...upload, ...update } : upload,
    );
  }

  function setConnection(online: boolean): void {
    connection = online ? "online" : "offline";
  }

  async function checkConnection(): Promise<void> {
    try {
      const response = await fetch("/api/health", {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(5000),
      });
      setConnection(response.ok);
    } catch {
      setConnection(false);
    }
  }

  async function loadConfig(): Promise<void> {
    try {
      const response = await fetch("/api/config", {
        headers: { Accept: "application/json" },
        signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) return;
      const config = (await response.json()) as { max_upload_bytes?: number };
      if (typeof config.max_upload_bytes === "number" && config.max_upload_bytes > 0) {
        maxUploadBytes = config.max_upload_bytes;
      }
    } catch {
      // Keep the local default when the config endpoint is unavailable.
    }
  }

  function fileExtension(file: File): string {
    return file.name.split(".").pop()?.toLowerCase() ?? "";
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024 * 1024) {
      return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function formatUploadLimit(bytes: number): string {
    if (bytes < 1_000_000) {
      return `${Math.max(1, Math.round(bytes / 1000))} KB`;
    }
    return `${(bytes / 1_000_000).toLocaleString(undefined, {
      maximumFractionDigits: 2,
    })} MB`;
  }

  function statusLabel(status: UploadStatus): string {
    const labels: Record<UploadStatus, string> = {
      uploading: "Uploading",
      queued: "Queued",
      processing: "Indexing",
      completed: "Ready",
      failed: "Failed",
    };
    return labels[status];
  }

  function waitForPollDelay(
    delay: number,
    signal: AbortSignal,
  ): Promise<boolean> {
    return new Promise((resolve) => {
      if (signal.aborted) {
        resolve(false);
        return;
      }

      const timer = setTimeout(() => {
        signal.removeEventListener("abort", cancel);
        resolve(true);
      }, delay);
      function cancel(): void {
        clearTimeout(timer);
        resolve(false);
      }
      signal.addEventListener("abort", cancel, { once: true });
    });
  }

  async function pollDocument(
    upload: Upload,
    controller: AbortController,
  ): Promise<void> {
    let retryDelay = 2500;
    let notifiedRetry = false;
    while (
      !controller.signal.aborted &&
      (upload.status === "queued" || upload.status === "processing")
    ) {
      if (!(await waitForPollDelay(retryDelay, controller.signal))) {
        return;
      }
      try {
        const response = await fetch(
          `/api/documents/${encodeURIComponent(upload.documentId!)}`,
          {
            headers: ingestionHeaders(),
            signal: AbortSignal.any([
              controller.signal,
              AbortSignal.timeout(10000),
            ]),
          },
        );
        if (controller.signal.aborted) return;
        if (!response.ok) {
          const detail = await responseError(response);
          if (controller.signal.aborted) return;
          if (
            response.status < 500 &&
            response.status !== 408 &&
            response.status !== 429
          ) {
            upload.status = "failed";
            upload.error = detail;
            updateUpload(upload.id, { status: "failed", error: detail });
            showToast(detail, true);
            break;
          }
          throw new Error(detail);
        }
        const result: {
          status: UploadStatus;
          chunks_indexed: number;
          error: string | null;
        } = await response.json();
        if (controller.signal.aborted) return;
        upload.status = result.status;
        updateUpload(upload.id, {
          status: result.status,
          chunksIndexed: result.chunks_indexed,
          error: result.error ?? undefined,
        });
        retryDelay = 2500;
        notifiedRetry = false;
      } catch (error) {
        if (controller.signal.aborted) return;
        const detail = errorMessage(error, "Could not check upload status.");
        if (!notifiedRetry) {
          showToast(`${detail} Retrying status checks.`, true);
          notifiedRetry = true;
        }
        retryDelay = Math.min(retryDelay * 2, 30000);
      }
    }

    if (upload.status === "completed") {
      uploadFeedback = `${upload.file.name} is ready to use.`;
      showToast(`${upload.file.name} is ready to use.`);
    } else if (upload.status === "failed") {
      uploadFeedback =
        upload.error || `${upload.file.name} could not be indexed.`;
      showToast(uploadFeedback, true);
    }
  }

  async function uploadFile(file: File): Promise<void> {
    const extension = fileExtension(file);
    if (!allowedExtensions.has(extension)) {
      uploadFeedback = "Choose a PDF, DOCX, PPTX, TXT, or Markdown file.";
      return;
    }
    if (file.size === 0) {
      uploadFeedback = "This file is empty.";
      return;
    }
    if (file.size > maxUploadBytes) {
      uploadFeedback =
        `Files must be ${formatUploadLimit(maxUploadBytes)} or smaller.`;
      return;
    }

    uploadFeedback = "";
    const upload: Upload = {
      id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`,
      file,
      status: "uploading",
    };
    uploads = [upload, ...uploads];
    uploadFeedback = `Uploading ${file.name}…`;

    const body = new FormData();
    body.append("file", file);
    const controller = new AbortController();
    activeUploadControllers.add(controller);

    try {
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        headers: ingestionHeaders(),
        body,
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const result: {
        document_id: string;
        status: UploadStatus;
      } = await response.json();
      if (controller.signal.aborted) return;
      upload.documentId = result.document_id;
      upload.status = result.status;
      updateUpload(upload.id, {
        documentId: result.document_id,
        status: result.status,
      });
      uploadFeedback = `${file.name} uploaded. Waiting for indexing…`;
      await pollDocument(upload, controller);
    } catch (error) {
      if (controller.signal.aborted) return;
      const detail = errorMessage(error, "The upload could not be completed.");
      upload.status = "failed";
      upload.error = detail;
      updateUpload(upload.id, { status: "failed", error: detail });
      uploadFeedback = detail;
      showToast(detail, true);
    } finally {
      activeUploadControllers.delete(controller);
    }
  }

  function handleFiles(files: FileList | File[] | null): void {
    if (!files) return;
    for (const file of Array.from(files)) {
      void uploadFile(file);
    }
  }

  function updateMessage(id: string, update: Partial<ChatMessage>): void {
    messages = messages.map((entry) =>
      entry.id === id ? { ...entry, ...update } : entry,
    );
  }

  function isGroundingStatus(value: unknown): value is GroundingStatus {
    return (
      value === "supported" ||
      value === "potential_drift" ||
      value === "unverified"
    );
  }

  async function sendMessage(prompt = message): Promise<void> {
    const question = prompt.trim();
    if (!question || isSending) return;

    message = "";
    if (messageInput) {
      messageInput.style.height = "auto";
    }
    isSending = true;
    const userId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-user`;
    const replyId =
      globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-assistant`;
    messages = [
      ...messages,
      { id: userId, role: "user", text: question },
      {
        id: replyId,
        role: "assistant",
        text: "",
        loading: true,
      },
    ];

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (chatKey.trim()) headers["X-API-Key"] = chatKey.trim();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers,
        body: JSON.stringify({ message: question, thread_id: threadId }),
      });
      if (!response.ok) {
        throw new Error(await responseError(response));
      }
      const result: {
        response: string;
        sources: string[];
        grounding_status?: unknown;
      } =
        await response.json();
      updateMessage(replyId, {
        text: result.response,
        sources: Array.isArray(result.sources)
          ? result.sources.map(String)
          : [],
        groundingStatus: isGroundingStatus(result.grounding_status)
          ? result.grounding_status
          : "unverified",
        loading: false,
      });
    } catch (error) {
      const detail = errorMessage(
        error,
        "The answer could not be generated.",
      );
      updateMessage(replyId, {
        text: detail,
        loading: false,
        error: true,
      });
      showToast(
        "Couldn't get an answer. Check the API connection and access key.",
        true,
      );
    } finally {
      isSending = false;
      messageInput?.focus();
    }
  }

  function handleMessageKeydown(event: KeyboardEvent): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  $effect(() => {
    if (messages.length > 0 && messagesElement) {
      messagesElement.scrollTop = messagesElement.scrollHeight;
    }
  });

  function resizeComposer(node: HTMLTextAreaElement): void {
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 155)}px`;
  }

  void checkConnection();
  void loadConfig();
</script>

<div class="workspace">
  <aside class="library" aria-label="Document library">
    <a class="brand" href="/" aria-label="Folio home">
      <span class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" fill="none">
          <path d="M8 6.5h11.5L25 12v13.5H8z" />
          <path d="M19 6.5V12h6M12 17h9M12 21h7" />
        </svg>
      </span>
      <span class="brand-name">folio<span class="brand-period">.</span></span>
    </a>

    <div class="library-heading">
      <div>
        <p class="eyebrow">YOUR WORKSPACE</p>
        <h1>Knowledge base</h1>
      </div>
      <span class="collection-badge" title="Single workspace collection">
        <span class="status-dot"></span>
        One collection
      </span>
    </div>

    <section class="upload-section" aria-labelledby="upload-title">
      <div class="section-heading">
        <h2 id="upload-title">Add documents</h2>
        <span class="section-count">01</span>
      </div>
      <label
        class:is-dragging={isDragging}
        class="dropzone"
        for="file-input"
        ondragover={(event) => {
          event.preventDefault();
          isDragging = true;
        }}
        ondragleave={() => (isDragging = false)}
        ondrop={(event) => {
          event.preventDefault();
          isDragging = false;
          handleFiles(event.dataTransfer?.files ?? null);
        }}
      >
        <input
          id="file-input"
          class="visually-hidden"
          type="file"
          accept=".pdf,.pptx,.docx,.txt,.md"
          multiple
          onchange={(event) => {
            const input = event.currentTarget;
            handleFiles(input.files);
            input.value = "";
          }}
        />
        <span class="upload-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 15V4m0 0L8 8m4-4 4 4M5 14v5h14v-5" />
          </svg>
        </span>
        <span class="dropzone-title">Drop files to begin</span>
        <span class="dropzone-caption">or <span>browse files</span></span>
        <span class="file-types">
          PDF, DOCX, PPTX, TXT, MD · up to {formatUploadLimit(maxUploadBytes)}
        </span>
      </label>
      <p class="upload-feedback" role="status" aria-live="polite">
        {uploadFeedback}
      </p>
    </section>

    <section class="documents-section" aria-labelledby="documents-title">
      <div class="section-heading">
        <h2 id="documents-title">Recent uploads</h2>
        <span class="section-count">{documentCount}</span>
      </div>
      {#if uploads.length === 0}
        <div class="document-list">
          <div class="empty-documents">
            <span class="empty-file-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M7 3.75h7l4 4v12.5H7z" />
                <path d="M14 3.75v4h4M10 13h5m-5 3.5h5" />
              </svg>
            </span>
            <p>Your library is waiting.</p>
            <span>Files you add in this tab will appear here.</span>
          </div>
        </div>
      {:else}
        <div class="document-list">
          {#each uploads as upload (upload.id)}
            <article class="document-item">
              <span class="file-kind">
                {fileExtension(upload.file).toUpperCase()}
              </span>
              <div class="document-meta">
                <div class="document-name" title={upload.file.name}>
                  {upload.file.name}
                </div>
                <div class="document-detail">
                  {#if upload.status === "completed" &&
                    upload.chunksIndexed != null}
                    {upload.chunksIndexed} chunks indexed
                  {:else if upload.status === "failed" && upload.error}
                    {upload.error}
                  {:else}
                    {formatBytes(upload.file.size)}
                  {/if}
                </div>
              </div>
              <span class="document-status" data-status={upload.status}>
                {statusLabel(upload.status)}
              </span>
            </article>
          {/each}
        </div>
      {/if}
    </section>

    <div class="sidebar-bottom">
      <details class="access-settings">
        <summary>
          <span class="settings-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" />
              <path
                d="m19.4 15 .1.1a1.7 1.7 0 1 1-2.4 2.4l-.1-.1a1.7 1.7 0 0 0-2.9 1.2v.2a1.7 1.7 0 1 1-3.4 0v-.2a1.7 1.7 0 0 0-2.9-1.2l-.1.1a1.7 1.7 0 1 1-2.4-2.4l.1-.1a1.7 1.7 0 0 0-1.2-2.9H4a1.7 1.7 0 1 1 0-3.4h.2a1.7 1.7 0 0 0 1.2-2.9l-.1-.1a1.7 1.7 0 1 1 2.4-2.4l.1.1a1.7 1.7 0 0 0 2.9-1.2V2a1.7 1.7 0 1 1 3.4 0v.2a1.7 1.7 0 0 0 2.9 1.2l.1-.1a1.7 1.7 0 1 1 2.4 2.4l-.1.1a1.7 1.7 0 0 0 1.2 2.9h.2a1.7 1.7 0 1 1 0 3.4h-.2a1.7 1.7 0 0 0-1.2 2.9Z"
              />
            </svg>
          </span>
          <span>Access keys</span>
          <span class="settings-chevron" aria-hidden="true">⌄</span>
        </summary>
        <div class="settings-fields">
          <label for="ingest-key">Document access key</label>
          <input
            id="ingest-key"
            type="password"
            bind:value={ingestKey}
            autocomplete="off"
            spellcheck="false"
          />
          <p>Local default shown; enter your deployment key if it differs.</p>
          <label for="chat-key">Chat access key <span>optional</span></label>
          <input
            id="chat-key"
            type="password"
            bind:value={chatKey}
            placeholder="Only if configured"
            autocomplete="off"
            spellcheck="false"
          />
          <p>Keys stay in this tab and are not saved.</p>
        </div>
      </details>
      <div class="connection">
        <span
          class="connection-indicator"
          class:is-online={connection === "online"}
          class:is-offline={connection === "offline"}
        ></span>
        <span>
          {connection === "online"
            ? "Workspace connected"
            : connection === "offline"
              ? "API unavailable"
              : "Connecting to your workspace…"}
        </span>
        <a href="/api/docs" target="_blank" rel="noreferrer">API</a>
      </div>
    </div>
  </aside>

  <main class="conversation">
    <header class="topbar">
      <div class="breadcrumb">
        <span>Workspace</span>
        <span class="breadcrumb-divider">/</span>
        <strong>Ask your documents</strong>
      </div>
      <div class="topbar-right">
        <span class="private-badge">
          <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect x="3" y="7" width="10" height="7" rx="1.5" />
            <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
          </svg>
          Single workspace
        </span>
        <span class="avatar" aria-label="You">Y</span>
      </div>
    </header>

    <div class="chat-shell">
      <div class="chat-heading">
        <div>
          <p class="eyebrow">YOUR KNOWLEDGE, IN CONVERSATION</p>
          <h2>What would you like to know?</h2>
          <p class="chat-subtitle">
            Ask a question and Folio will find the relevant details in your
            documents.
          </p>
        </div>
        <span class="answer-mode">
          <span class="answer-mode-dot"></span>
          Answers with sources
        </span>
      </div>

      <section
        class="messages"
        bind:this={messagesElement}
        aria-label="Conversation"
        aria-live="polite"
      >
        {#if messages.length === 0}
          <div class="welcome-card">
            <div class="welcome-symbol" aria-hidden="true">
              <svg viewBox="0 0 32 32" fill="none">
                <path d="M8 7h11l5 5v13H8z" />
                <path d="M19 7v5h5M12 17h8m-8 4h6" />
                <path d="M4.5 11.5V4.5h11" />
              </svg>
            </div>
            <p class="welcome-kicker">A fresh perspective starts here</p>
            <h3>Your documents have the answers.</h3>
            <p class="welcome-copy">
              Add a document to your library, then ask a question in your own
              words. Answers include references back to their sources.
            </p>
            <div class="suggestions" aria-label="Example questions">
              {#each suggestions as suggestion}
                <button
                  type="button"
                  class="suggestion"
                  onclick={() => void sendMessage(suggestion)}
                  disabled={isSending}
                >
                  <span aria-hidden="true">↗</span>
                  {suggestion === suggestions[0]
                    ? "Summarize the key points"
                    : suggestion === suggestions[1]
                      ? "Find dates and deadlines"
                      : "What should I know first?"}
                </button>
              {/each}
            </div>
          </div>
        {:else}
          {#each messages as entry (entry.id)}
            <article
              class="message-row"
              class:user={entry.role === "user"}
              class:is-error={entry.error}
            >
              <span class="message-avatar" aria-hidden="true">
                {entry.role === "user" ? "Y" : "F"}
              </span>
              <div class="message-content">
                <p class="message-label">
                  {entry.role === "user" ? "You" : "Folio"}
                </p>
                <div
                  class="message-bubble"
                  aria-label={entry.loading ? "Finding an answer" : undefined}
                >
                  {#if entry.loading}
                    <span class="typing-indicator">
                      <span></span><span></span><span></span>
                    </span>
                  {:else}
                    {entry.text}
                  {/if}
                </div>
                {#if entry.role === "assistant" && !entry.loading && !entry.error}
                  <p
                    class="grounding-status"
                    data-status={entry.groundingStatus ?? "unverified"}
                    aria-label="Answer evidence check"
                  >
                    {entry.groundingStatus === "supported"
                      ? "Evidence check: appears supported"
                      : entry.groundingStatus === "potential_drift"
                        ? "Document answer drift detected; using general knowledge"
                        : "Not verified against documents"}
                  </p>
                {/if}
                {#if entry.sources?.length}
                  <div class="message-sources">
                    {#each entry.sources as source}
                      <span class="source-label">
                        <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                          <path
                            d="M4 2.5h5l3 3v8H4zM9 2.5v3h3M6 8.5h4M6 11h4"
                          />
                        </svg>
                        {source}
                      </span>
                    {/each}
                  </div>
                {/if}
              </div>
            </article>
          {/each}
        {/if}
      </section>

      <form
        class="composer"
        onsubmit={(event) => {
          event.preventDefault();
          void sendMessage();
        }}
      >
        <label class="visually-hidden" for="message-input">
          Ask your documents
        </label>
        <textarea
          id="message-input"
          bind:this={messageInput}
          bind:value={message}
          rows="1"
          maxlength="10000"
          placeholder="Ask anything about your documents…"
          required
          disabled={isSending}
          onkeydown={handleMessageKeydown}
          oninput={(event) => resizeComposer(event.currentTarget)}
        ></textarea>
        <div class="composer-footer">
          <p>
            <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <circle cx="8" cy="8" r="6" />
              <path d="M8 7v4m0-6h.01" />
            </svg>
            Answers may be imperfect. Verify important details in the sources.
          </p>
          <button class="send-button" type="submit" disabled={isSending}>
            <span>{isSending ? "Thinking" : "Send"}</span>
            <svg viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="M4 10h12m-5-5 5 5-5 5" />
            </svg>
          </button>
        </div>
      </form>
      <p class="conversation-note">
        Each question is answered independently. Conversation history is only
        shown in this tab.
      </p>
    </div>
  </main>
</div>

{#if toast}
  <div
    class="toast"
    class:is-visible={Boolean(toast)}
    class:is-error={toastError}
    role="status"
    aria-live="polite"
  >
    {toast}
  </div>
{/if}

<svelte:window
  onfocus={() => void checkConnection()}
  onbeforeunload={() => clearInterval(connectionTimer)}
/>
