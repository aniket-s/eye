import './styles/app.css';
import './styles/components.css';

import { DictionaryPage } from './pages/dictionary.js';
import { TranslatorPage } from './pages/translator.js';
import { Router } from './router.js';
import { SideMenu } from './ui/sideMenu.js';

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
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootstrap, { once: true });
} else {
  bootstrap();
}
