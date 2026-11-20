import "@chatui/core/dist/index.css";
import { useEffect, useRef, useState } from "react";
import { pipeline, TextGenerationPipeline } from "@huggingface/transformers";
import ChatModule, {
  Bubble as BubbleModule,
  useMessages,
  type MessageProps,
} from "@chatui/core";

import botImage from "/botImage.jpg";

const Chat =
  (ChatModule as unknown as { default?: typeof ChatModule }).default ??
  ChatModule;
const Bubble =
  (BubbleModule as unknown as { default?: typeof BubbleModule }).default ??
  BubbleModule;

const botUser = {
  avatar: botImage,
};

const userContextPrompt = [
  "You are Islem's personal assistant chatbot running locally in the browser",
  "Key facts about Amrani Younes Islem:",
  "- Name: Amrani Islem, 20 years old, born on July 13th 2006, living in Oran, Algeria.",
  "- Education: L3 ISIL Computer Science student at USTO (Université des Sciences et de la Technologie d'Oran - Mohamed Boudiaf), expected graduation 2027.",
  "- Programming languages i know: C, C++, Rust, Java, HTML, CSS, Javascript, Typescript, Python, PHP, C#, GDScript, Zig, Go.",
  "- Games i played and liked: Resident evil 2, 3 and 7 Biohazard, Dark souls Remastered, Brawlhalla, Minecraft, Stardew Valley, Call of duty: World at war, Clash Royale..",
  "- Height: 180cm, Weight: 70kg",
  "- My shoe size is 43 eu and i wear size M for shirts and pants.",
  "- Languages i speak: English, Arabic and French.",
  "- Games i made: Minio, Pixelheli",
  "- My socials are: Instagram: isle26m, Facebook: Islem Amrani, Chess.com: islemtal, Github: isle26m",
  "- Favorite color: Blue",
  "- Favorite fruit: Grapes",
  "- I started coding since I was 13, the first thing I made was a Discord bot using Discord.js tho before I used to be a Video Editor for Ascends using both Sony Vegas Pro and Adobe After Effects",
  "- I fell in love with programming after meeting a bunch of online friends in a community called Fossbay and I then became a member in it, one of my biggest inspirations is Codergopher.",
  "- Favorite band: Evanescence.",
  "- Relationship status: Currently single and not looking to date anyone.",
  "- Tech stack & interests: Game developement using Unity and Godot, Pixel art, React, Vite, Tailwind CSS, Node.js, Express, PostgreSQL, Socket.IO, GSAP, Framer Motion, Anime.js, local LLMs via Ollama and Jan, client-side ML (MediaPipe, ONNX Runtime Web, Transformers.js), Arch/CachyOS Linux with Wayland tiling window managers, gym, weightlifting, somewhat good at chess, music production using FL Studio, rock/metal music..",
  "If asked if i miss someone, say yes a lot but I have to force myself to avoid them, don't mention this if not asked.",
  "Answer concisely, accurately, and helpfully based on these facts when asked about Islem or general questions.",
].join("\n");

const initialMessages: Omit<MessageProps, "_id">[] = [
  {
    type: "text",
    content: {
      text: "Hello! Welcome to Islem's Chatbot. Please wait while the AI model is loading in your browser..",
    },
    user: botUser,
  },
];

export default function App() {
  const { messages, appendMsg, updateMsg, deleteMsg } =
    useMessages(initialMessages);
  const generatorRef = useRef<TextGenerationPipeline | null>(null);
  const [modelReady, setModelReady] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [, setIsTyping] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadModel() {
      try {
        const pipe = await pipeline(
          "text-generation",
          "HuggingFaceTB/SmolLM2-135M-Instruct",
          {
            dtype: "q4",
          },
        );

        if (!active) return;

        generatorRef.current = pipe;
        setModelReady(true);

        appendMsg({
          type: "text",
          content: {
            text: "The AI model is ready !",
          },
          position: "left",
          user: botUser,
        });
      } catch {
        if (!active) return;

        appendMsg({
          type: "text",
          content: {
            text: "Failed to load the AI model. Check your connection or browser WebGPU/WASM support.",
          },
          position: "left",
          user: botUser,
        });
      }
    }

    loadModel();

    return () => {
      active = false;
    };
  }, [appendMsg]);

  async function triggerBotReply(userText: string) {
    if (isGenerating) return;

    const loadingId = `loading-${Date.now()}`;
    setIsGenerating(true);
    setIsTyping(true);

    appendMsg({
      _id: loadingId,
      type: "text",
      content: { text: "Thinking..." },
      position: "left",
      user: botUser,
    });

    await new Promise((resolve) => window.requestAnimationFrame(resolve));

    try {
      let replyText =
        "Model is still loading, please wait a moment and try again.";

      if (generatorRef.current && modelReady) {
        const chatMessages = [
          { role: "system", content: userContextPrompt },
          { role: "user", content: userText },
        ];

        const output = await generatorRef.current(chatMessages, {
          max_new_tokens: 160,
          temperature: 0.7,
          top_p: 0.9,
          do_sample: true,
        });

        const firstItem = Array.isArray(output) ? output[0] : output;
        const generated =
          typeof firstItem === "object" &&
          firstItem !== null &&
          "generated_text" in firstItem
            ? firstItem.generated_text
            : "";

        if (Array.isArray(generated)) {
          const lastAssistant = [...generated]
            .reverse()
            .find((m) => m?.role === "assistant");
          replyText = String(lastAssistant?.content ?? "").trim() || replyText;
        } else if (typeof generated === "string") {
          replyText = generated.trim() || replyText;
        }
      }

      setIsTyping(false);
      updateMsg(loadingId, {
        type: "text",
        content: { text: replyText },
        position: "left",
        user: botUser,
      });
    } catch {
      setIsTyping(false);
      deleteMsg(loadingId);
      appendMsg({
        type: "text",
        content: {
          text: "Something went wrong while generating a response with the AI model.",
        },
        position: "left",
        user: botUser,
      });
    } finally {
      setIsGenerating(false);
    }
  }

  function handleSend(type: string, val: string) {
    if (isGenerating) return;

    const text = String(val ?? "").trim();
    if (!text) return;

    if (type === "text" || !type) {
      appendMsg({
        type: "text",
        content: { text },
        position: "right",
      });
      triggerBotReply(text);
    }
  }

  function renderMessageContent(msg: MessageProps) {
    const { type, content } = msg;

    if (type === "text" && content?.text) {
      return <Bubble content={content.text} />;
    }

    return null;
  }

  const inputPlaceholder = !modelReady
    ? "Loading AI model..."
    : isGenerating
      ? "Generating response..."
      : "Type a message...";

  return (
    <div style={{ height: "100vh", width: "100vw" }}>
      <Chat
        locale="en-US"
        navbar={{ title: "Islem's Chatbot" }}
        messages={messages}
        renderMessageContent={renderMessageContent}
        onSend={handleSend}
        placeholder={inputPlaceholder}
        inputOptions={{
          placeholder: inputPlaceholder,
        }}
        toolbar={[]}
        loadMoreText="Load earlier messages"
        wideBreakpoint="600px"
      />
    </div>
  );
}
