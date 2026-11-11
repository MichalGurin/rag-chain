import { mount } from "svelte";
import App from "./App.svelte";
import "./app.css";

const target = document.querySelector<HTMLElement>("#app");
if (!target) {
  throw new Error("Could not find the frontend mount point.");
}

mount(App, { target });
