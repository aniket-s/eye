// Self-hosted font. Previously loaded from fonts.googleapis.com, which meant a
// third-party request from an app whose premise is that nothing leaves the device —
// and a hard failure on restricted networks (docs/AUDIT.md, A10). Vite subsets and
// hashes only the weights actually referenced.
import '@fontsource-variable/inter/index.css';

import './styles/app.css';
import './styles/components.css';

import { DictionaryPage } from './pages/dictionary.js';
import { TranslatorPage } from './pages/translator.js';
import { Router } from './router.js';
import { SideMenu } from './ui/sideMenu.js';
import { registerServiceWorker, watchConnectivity } from './offline.js';

/**
 * Application entry point.
 *
 * Wires the pages together and kicks off the model download. Nothing here does any
 * recognition work — that lives in `@mudrapragyan/core` so it can be tested in Node.
 */
function bootstrap(): void {
  const router = new Router();
  const sideMenu = new SideMenu();
  const dictionary = new DictionaryPage();
  const translator = new TranslatorPage();

  sideMenu.start();
  dictionary.start();
  translator.start();

  router.subscribe((page) => {
    sideMenu.close();
    if (page === 'dictionary') dictionary.build();
    // Release the camera when it is not on screen. v1 left it running in the
    // background on every other page, which kept the webcam light on.
    if (page !== 'translator') void translator.stopCamera();
  });
  router.start();

  // The app genuinely keeps working offline once cached, so say so rather than letting
  // the user assume it has broken.
  const banner = document.getElementById('offlineBanner');
  watchConnectivity((online) => {
    if (banner !== null) banner.hidden = online;
  });
  registerServiceWorker();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap, { once: true });
} else {
  bootstrap();
}
