# Privacy

**Your camera video never leaves your device.**

This is not a policy promise — it is how the application is built. There is no server to send
video to. MudraPragyan.AI is a static website; once the page has loaded, the recognition models
run entirely inside your browser.

## What happens to your camera feed

1. You press **Start Camera**. Your browser asks your permission.
2. Frames go to a hand-tracking model running in your browser tab.
3. That model produces 21 numeric hand positions per frame — still in your browser.
4. A classifier turns those positions into a letter — still in your browser.
5. The frame is discarded. Nothing is recorded, uploaded, or stored.

## What we collect

Nothing. There is no analytics, no tracking, no cookies, no account, and no server that receives
anything you do. The sentence you build exists only in the page's memory and is gone when you
close the tab.

## What the page does download

Static files: the application code, the recognition models, and images. These are ordinary file
downloads, the same as any website.

The page currently also requests a webfont from `fonts.googleapis.com`, which means Google's
servers see the request. **This is the one external request the app makes, and we are removing
it** by self-hosting the font (tracked as A10 in `docs/AUDIT.md`).

## Permissions

The app asks for **camera access only**, and only when you press Start. It never asks for a
microphone, location, or files. Camera access is released when you navigate away from the
Translator, and your browser's camera indicator will switch off.

Text-to-speech uses your browser's built-in speech synthesis. Depending on your browser and
operating system, that feature may process the text on the device or through the OS vendor's
service — that behaviour belongs to your browser, not to this app.

## Verifying this yourself

You do not have to take our word for it:

- Open your browser's developer tools, go to the **Network** tab, and use the app. You will see
  static file downloads and nothing else.
- Disconnect from the internet after the page loads. Recognition keeps working.
- Read the source. It is open, and the recognition code is in `packages/core`.

## Contact

Found a privacy problem? Open an issue in the repository. Please do not include personal data.
