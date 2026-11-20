# Preview containment

**The proposed sandbox and CSP do not prevent a standalone HTML preview from sending repository text through its own navigation.**
There is no HTML preview route in the web server.

## Reproduce

Install the project's Playwright browsers, then run:

```sh
uv run python bench/web_preview_containment.py --output /tmp/preview-containment.json
```

The probe uses a local UI shell, a preview document, a text asset and separate HTTP and UDP listeners.
It does not run the agent6 web server or use real credentials.
Every preview document and asset carries `sandbox allow-scripts` in its CSP and `Referrer-Policy: no-referrer`.
The frame's sandbox attribute is exactly `allow-scripts`.
The shell's `frame-src` permits only the preview path.
The preview's policy permits only that path for resources, with inline script and style, and denies forms, plugins, bases and child frames.
The JSON report records the full policy for each probe.

Each script reads `repository-sample` from the preview asset before trying its exit.
The probe records real TCP connections, HTTP paths, UDP datagrams and browser console messages, without request interception.
It observes each case for one second, or three seconds for WebRTC.
A received connection, API request or storage read makes the command fail after recording all cases.

## Observations

Chromium 156.0.8078.4 and Firefox 157.0 each ran 16 probes inside the frame and directly in a tab (64 cases).

| Probe | Chromium frame | Chromium tab | Firefox frame | Firefox tab |
| --- | --- | --- | --- | --- |
| Self navigation | No traffic | Received repository text | No traffic | Received repository text |
| Top navigation | No traffic | Received repository text | No traffic | Received repository text |
| Fetch, XMLHttpRequest, image, script, stylesheet, font, form, pop-up, WebSocket, prefetch, preconnect, API | No traffic | No traffic | No traffic | No traffic |
| UI storage | SecurityError | SecurityError | SecurityError | SecurityError |
| WebRTC | No traffic observed | No traffic observed | No traffic observed | No traffic observed |

The HTTP listener received `/self-navigation?sample=repository-sample` and `/top-navigation?sample=repository-sample` from each browser's standalone tab.
In a standalone tab, `top` is the preview itself.
Firefox also requested the destination's favicon.
The probe exited 1 with:

```text
preview containment failed: chromium/tab/self-navigation, chromium/tab/top-navigation, firefox/tab/self-navigation, firefox/tab/top-navigation
```

The parent policy blocked the framed page's self navigation.
The sandbox stopped the framed page from navigating its parent and kept both contexts from reading UI storage.
Neither policy stopped a standalone preview from navigating itself.

A second run of navigation, storage and WebRTC probes added `navigate-to 'none'; webrtc 'block'` to the CSP (16 cases).
Both browsers still sent repository text through standalone navigation.
Chromium reported both directives as unrecognized.

**WebRTC is not certified.** The probe ran with only a loopback interface and received no UDP datagrams.
That does not prove WebRTC containment on a networked machine.
The observed navigation exit is sufficient to reject the policy without relying on the quiet probes.

## Boundary

The script keeps the no-egress assertion failing rather than marking the observed exits as expected passes.
Arbitrary HTML scripts with a directly executable preview address need a boundary beyond these tested browser policies.
Disabling repository scripts or refusing standalone execution would change what the preview supports.
Neither alternative is implemented.
