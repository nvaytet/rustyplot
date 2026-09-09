// anywidget frontend. Deliberately thin: it forwards DOM events into the wasm
// module and does no plotting maths of its own.

/** The wasm-bindgen glue is shipped as source text, so import it as a module at runtime. */
async function importModuleFromSource(source) {
    const blob = new Blob([source], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    try {
        return await import(/* webpackIgnore: true */ /* @vite-ignore */ url);
    } catch (err) {
        // Some notebook hosts block blob: imports under CSP; data: URLs usually survive.
        const dataUrl =
            "data:text/javascript;base64," +
            btoa(String.fromCharCode(...new TextEncoder().encode(source)));
        return await import(/* webpackIgnore: true */ /* @vite-ignore */ dataUrl);
    } finally {
        URL.revokeObjectURL(url);
    }
}

function asFloat32(value) {
    if (!value) return new Float32Array(0);
    const view = value instanceof DataView ? value : new DataView(value.buffer ?? value);
    return new Float32Array(view.buffer, view.byteOffset, view.byteLength / 4);
}

function showError(el, message) {
    const box = document.createElement("pre");
    box.className = "rustyplot-error";
    box.textContent = message;
    el.appendChild(box);
}

/**
 * Decide which backend to use before the real canvas is touched.
 *
 * `navigator.gpu` existing does not mean WebGPU works: on many Linux setups an
 * adapter request still comes back empty. Probing here, on a throwaway canvas,
 * keeps the real one unbound so the WebGL2 fallback stays available.
 */
async function detectBackend() {
    if (navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) return { useWebgpu: true, name: "WebGPU" };
        } catch (err) {
            console.warn("rustyplot: WebGPU probe failed, falling back to WebGL2", err);
        }
    }
    const probe = document.createElement("canvas");
    const gl = probe.getContext("webgl2");
    if (gl) {
        gl.getExtension("WEBGL_lose_context")?.loseContext();
        return { useWebgpu: false, name: "WebGL2" };
    }
    return null;
}

async function render({ model, el }) {
    el.classList.add("rustyplot-widget");

    const container = document.createElement("div");
    container.className = "rustyplot-container";
    container.style.width = `${model.get("width")}px`;
    container.style.height = `${model.get("height")}px`;

    const canvas = document.createElement("canvas");
    canvas.className = "rustyplot-canvas";
    container.appendChild(canvas);
    el.appendChild(container);

    const dpr = () => window.devicePixelRatio || 1;
    const sizeCanvas = () => {
        const rect = container.getBoundingClientRect();
        const w = Math.max(1, Math.round(rect.width * dpr()));
        const h = Math.max(1, Math.round(rect.height * dpr()));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
            return true;
        }
        return false;
    };
    sizeCanvas();

    let plot;
    let naxes;
    try {
        const backend = await detectBackend();
        if (!backend) {
            throw new Error(
                "this browser exposes neither a working WebGPU adapter nor WebGL2"
            );
        }
        const wasm = await importModuleFromSource(model.get("_wasm_js"));
        const binary = model.get("_wasm_binary");
        await wasm.default({ module_or_path: binary.buffer ?? binary });
        const nrows = model.get("nrows");
        const ncols = model.get("ncols");
        naxes = nrows * ncols;
        plot = await wasm.Plot.attach(canvas, backend.useWebgpu, nrows, ncols);
        console.debug(`rustyplot: rendering with ${backend.name}`);
    } catch (err) {
        showError(
            el,
            `rustyplot could not start the GPU renderer.\n\n${err}\n\n` +
            `navigator.gpu: ${navigator.gpu ? "present" : "absent"}\n` +
            `webgl2: ${document.createElement("canvas").getContext("webgl2") ? "available" : "unavailable"}`
        );
        return;
    }

    let frameRequested = false;
    const draw = () => {
        frameRequested = false;
        try {
            plot.draw();
        } catch (err) {
            console.error("rustyplot draw failed", err);
        }
    };
    const requestDraw = () => {
        if (!frameRequested) {
            frameRequested = true;
            requestAnimationFrame(draw);
        }
    };

    /** Push one axes' point data from its trait lists into the wasm `Plot`,
     * then autoscale that axes to fit (matching the historical behaviour
     * that a fresh `scatter()` call always frames the new data). */
    const pushAxis = (i) => {
        const x = asFloat32(model.get("_x")[i]);
        if (x.length === 0) return;
        const y = asFloat32(model.get("_y")[i]);
        const size = asFloat32(model.get("_size")[i]);
        const color = asFloat32(model.get("_color")[i]);
        try {
            plot.set_scatter(i, x, y, size, color);
            plot.autoscale(i);
        } catch (err) {
            console.error("rustyplot could not accept the data", err);
        }
    };

    const applyLabels = () => {
        const titles = model.get("titles");
        const xlabels = model.get("xlabels");
        const ylabels = model.get("ylabels");
        for (let i = 0; i < naxes; i++) {
            plot.set_title(i, titles[i] ?? "");
            plot.set_xlabel(i, xlabels[i] ?? "");
            plot.set_ylabel(i, ylabels[i] ?? "");
        }
    };

    // Guards against the frontend's own `syncView` write below feeding back
    // into this handler: `model.set()` fires its change event synchronously.
    let applyingLocalView = false;
    const applyViewFromModel = () => {
        if (applyingLocalView) return;
        const view = model.get("view");
        for (let i = 0; i < naxes; i++) {
            const [xMin, xMax, yMin, yMax] = view[i];
            plot.set_view(i, xMin, xMax, yMin, yMax);
        }
    };

    // Initial state: every trait is replayed in full when a widget is first
    // displayed (unlike a custom message, which needs a live browser-side
    // model to receive it), so this covers data set before `fig` was shown.
    const bg = model.get("background");
    plot.set_background(bg[0], bg[1], bg[2], bg[3]);
    applyLabels();
    applyViewFromModel();
    for (let i = 0; i < naxes; i++) pushAxis(i);
    requestDraw();

    // --- events -------------------------------------------------------------
    // Pointer coordinates are converted to physical pixels because that is the
    // space the Rust side works in.
    const toPixels = (event) => {
        const rect = canvas.getBoundingClientRect();
        const scale = dpr();
        return [(event.clientX - rect.left) * scale, (event.clientY - rect.top) * scale];
    };

    let viewSyncTimer = null;
    const syncView = () => {
        clearTimeout(viewSyncTimer);
        viewSyncTimer = setTimeout(() => {
            applyingLocalView = true;
            try {
                const view = [];
                for (let i = 0; i < naxes; i++) view.push(Array.from(plot.view(i)));
                model.set("view", view);
                model.save_changes();
            } finally {
                applyingLocalView = false;
            }
        }, 120);
    };

    const onPointerDown = (event) => {
        canvas.setPointerCapture(event.pointerId);
        plot.pointer_down(...toPixels(event));
    };

    const onPointerMove = (event) => {
        if (plot.pointer_move(...toPixels(event))) requestDraw();
    };

    const onPointerUp = (event) => {
        const [px, py] = toPixels(event);
        const wasClick = plot.pointer_up();
        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
        if (wasClick) {
            const hit = plot.pick(px, py);
            const dataAt = plot.data_at(px, py);
            model.send({
                type: "click",
                axes: hit ? hit[0] : null,
                x: dataAt ? dataAt[0] : null,
                y: dataAt ? dataAt[1] : null,
                index: hit ? hit[2] : null,
                point_x: hit ? hit[3] : null,
                point_y: hit ? hit[4] : null,
            });
        } else {
            syncView();
        }
    };

    const onWheel = (event) => {
        event.preventDefault();
        const [px, py] = toPixels(event);
        // deltaMode 1 is lines, 2 is pages; normalise both to pixels.
        const scale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 100 : 1;
        plot.wheel(px, py, event.deltaY * scale);
        requestDraw();
        syncView();
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });

    const observer = new ResizeObserver(() => {
        if (sizeCanvas()) {
            plot.resize(canvas.width, canvas.height);
            requestDraw();
        }
    });
    observer.observe(container);

    const onDataChange = () => {
        // `_dirty_axes` names every axes touched since it was last consumed
        // (plural: a single `with fig.hold():` block can batch `scatter()`
        // calls to more than one axes into this one notification). Reset it
        // once applied so a later, unrelated update does not re-touch and
        // re-autoscale axes that were already pushed.
        const dirty = model.get("_dirty_axes");
        if (dirty.length === 0) return;
        for (const i of dirty) pushAxis(i);
        model.set("_dirty_axes", []);
        model.save_changes();
        requestDraw();
    };
    model.on("change:_revision", onDataChange);

    const onLabelsChange = () => {
        applyLabels();
        requestDraw();
    };
    model.on("change:titles", onLabelsChange);
    model.on("change:xlabels", onLabelsChange);
    model.on("change:ylabels", onLabelsChange);

    const onViewChange = () => {
        applyViewFromModel();
        requestDraw();
    };
    model.on("change:view", onViewChange);

    const onSizeChange = () => {
        container.style.width = `${model.get("width")}px`;
        container.style.height = `${model.get("height")}px`;
    };
    model.on("change:width", onSizeChange);
    model.on("change:height", onSizeChange);

    // Returned teardown stops the render loop and frees the GPU surface when the
    // cell is re-executed; without it every re-run leaks a WebGPU context.
    return () => {
        observer.disconnect();
        clearTimeout(viewSyncTimer);
        canvas.removeEventListener("pointerdown", onPointerDown);
        canvas.removeEventListener("pointermove", onPointerMove);
        canvas.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("pointercancel", onPointerUp);
        canvas.removeEventListener("wheel", onWheel);
        model.off("change:_revision", onDataChange);
        model.off("change:titles", onLabelsChange);
        model.off("change:xlabels", onLabelsChange);
        model.off("change:ylabels", onLabelsChange);
        model.off("change:view", onViewChange);
        model.off("change:width", onSizeChange);
        model.off("change:height", onSizeChange);
        plot.free();
    };
}

export default { render };
