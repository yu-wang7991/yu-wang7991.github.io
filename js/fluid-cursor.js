/**
 * WebGL Fluid Simulation - FluidCursor
 * Based on PavelDoGreat's WebGL-Fluid-Simulation
 * Adapted for Hexo blog integration
 */
(function () {
  'use strict';

  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:block;position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:9999';
  document.body.appendChild(canvas);

  const params = {
    simResolution: 128,
    dyeResolution: 1024,
    densityDissipation: 3.5,
    velocityDissipation: 2,
    pressure: 0.1,
    pressureIterations: 20,
    curl: 3,
    splatRadius: 0.2,
    splatForce: 6000,
    shading: true,
    colorUpdateSpeed: 10,
    transparent: true
  };

  let gl, ext;
  let simWidth, simHeight, dyeWidth, dyeHeight;
  let density, velocity, divergence, curl, pressure;
  let ditheringTexture;
  let lastTime = 0;
  let colorTime = 0;
  let animationStarted = false;

  const pointers = [pointer()];

  function pointer() {
    return {
      id: -1, texcoordX: 0, texcoordY: 0,
      prevTexcoordX: 0, prevTexcoordY: 0,
      deltaX: 0, deltaY: 0,
      down: false, moved: false,
      color: { r: 0, g: 0, b: 0 }
    };
  }

  // --- WebGL Helpers ---
  function getWebGLContext() {
    const c = canvas;
    c.width = window.innerWidth;
    c.height = window.innerHeight;
    const ctx = c.getContext('webgl2', { alpha: true, desynchronized: true });
    if (!ctx) { console.warn('WebGL 2 not supported, fluid cursor disabled'); return null; }
    const e = {
      formatRGBA: getSupportedFormat(ctx, ctx.RGBA, ctx.RGBA),
      formatRG: getSupportedFormat(ctx, ctx.RG, ctx.RGBA),
      formatR: getSupportedFormat(ctx, ctx.RED, ctx.RGBA),
      halfFloatTexType: ctx.HALF_FLOAT,
      supportLinearFiltering: ctx.getExtension('OES_texture_float_linear') ? true : false
    };
    if (!e.formatRG) e.formatRG = e.formatR;
    if (!e.formatRGBA) { console.warn('WebGL fluid: RGBA format not supported'); return null; }
    return { gl: ctx, ext: e };
  }

  function getSupportedFormat(gl, internalFormat, format) {
    const halfFloat = gl.HALF_FLOAT;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, halfFloat, null);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.deleteTexture(tex);
    return status === gl.FRAMEBUFFER_COMPLETE ? { internalFormat, format } : null;
  }

  // --- Shader System ---
  class Program {
    constructor(vertexSrc, fragmentSrc) {
      this.uniforms = {};
      this.program = createProgram(vertexSrc, fragmentSrc);
      this.uniforms = getUniforms(this.program);
    }
    bind() { gl.useProgram(this.program); }
  }

  function compileShader(type, source) {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  }

  function createProgram(vSrc, fSrc) {
    const p = gl.createProgram();
    const vs = compileShader(gl.VERTEX_SHADER, vSrc);
    const fs = compileShader(gl.FRAGMENT_SHADER, fSrc);
    if (!vs || !fs) return p;
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p));
    return p;
  }

  function getUniforms(program) {
    const u = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(program, i);
      u[info.name] = gl.getUniformLocation(program, info.name);
    }
    return u;
  }

  // --- Framebuffer ---
  class FBO {
    constructor(w, h, internalFormat, format, type, filtering) {
      this.width = w; this.height = h;
      this.texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filtering);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filtering);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);
      this.fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture, 0);
      gl.viewport(0, 0, w, h);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
  }

  function createDoubleFBO(w, h, internalFormat, format, type, filtering) {
    return {
      width: w, height: h,
      read: new FBO(w, h, internalFormat, format, type, filtering),
      write: new FBO(w, h, internalFormat, format, type, filtering),
      swap() { const t = this.read; this.read = this.write; this.write = t; }
    };
  }

  function resizeFBO(target, w, h, internalFormat, format, type, filtering) {
    const newFBO = createDoubleFBO(w, h, internalFormat, format, type, filtering);
    return newFBO;
  }

  // --- Shaders ---
  const baseVertexSrc = `#version 300 es
    precision highp float;
    in vec2 aPosition;
    out vec2 vUv;
    out vec2 vL; out vec2 vR; out vec2 vT; out vec2 vB;
    uniform vec2 texelSize;
    void main(){
      vUv = aPosition * 0.5 + 0.5;
      vL = vUv - vec2(texelSize.x, 0.0);
      vR = vUv + vec2(texelSize.x, 0.0);
      vT = vUv + vec2(0.0, texelSize.y);
      vB = vUv - vec2(0.0, texelSize.y);
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }`;

  const clearSrc = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    uniform sampler2D uTexture;
    uniform float value;
    out vec4 fragColor;
    void main(){ fragColor = value * texture(uTexture, vUv); }`;

  const displaySrc = `#version 300 es
    precision highp float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uTexture;
    uniform vec2 texelSize;
    out vec4 fragColor;
    void main(){
      vec3 c = texture(uTexture, vUv).rgb;
      #ifdef SHADING
        vec3 lc = texture(uTexture, vL).rgb;
        vec3 rc = texture(uTexture, vR).rgb;
        vec3 tc = texture(uTexture, vT).rgb;
        vec3 bc = texture(uTexture, vB).rgb;
        float dx = length(rc) - length(lc);
        float dy = length(tc) - length(bc);
        vec3 n = normalize(vec3(dx, dy, length(texelSize)));
        float ambient = clamp(0.7 + dot(n, vec3(0.0, 0.0, 1.0)), 0.7, 1.0);
        c *= ambient;
      #endif
      float a = max(c.r, max(c.g, c.b));
      fragColor = vec4(c, a);
    }`;

  const splatSrc = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D uTarget;
    uniform float aspectRatio;
    uniform vec3 color;
    uniform vec2 point;
    uniform float radius;
    out vec4 fragColor;
    void main(){
      vec2 p = vUv - point.xy;
      p.x *= aspectRatio;
      vec3 splat = exp(-dot(p,p) / radius) * color;
      vec3 base = texture(uTarget, vUv).xyz;
      fragColor = vec4(base + splat, 1.0);
    }`;

  const advectionSrc = `#version 300 es
    precision highp float;
    in vec2 vUv;
    uniform sampler2D uVelocity;
    uniform sampler2D uSource;
    uniform vec2 texelSize;
    uniform float dt;
    uniform float dissipation;
    out vec4 fragColor;
    void main(){
      vec2 coord = vUv - dt * texture(uVelocity, vUv).xy * texelSize;
      vec4 result = dissipation * texture(uSource, coord);
      fragColor = result;
    }`;

  const curlSrc = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uVelocity;
    out vec4 fragColor;
    void main(){
      float L = texture(uVelocity, vL).y;
      float R = texture(uVelocity, vR).y;
      float T = texture(uVelocity, vT).x;
      float B = texture(uVelocity, vB).x;
      float vorticity = R - L - T + B;
      fragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);
    }`;

  const vorticitySrc = `#version 300 es
    precision highp float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uVelocity;
    uniform sampler2D uCurl;
    uniform float curl;
    uniform float dt;
    out vec4 fragColor;
    void main(){
      float L = texture(uCurl, vL).x;
      float R = texture(uCurl, vR).x;
      float T = texture(uCurl, vT).x;
      float B = texture(uCurl, vB).x;
      float C = texture(uCurl, vUv).x;
      vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
      force /= length(force) + 0.0001;
      force *= curl * C;
      force.y *= -1.0;
      vec2 velocity = texture(uVelocity, vUv).xy;
      velocity += force * dt;
      fragColor = vec4(velocity, 0.0, 1.0);
    }`;

  const divergenceSrc = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uVelocity;
    out vec4 fragColor;
    void main(){
      float L = texture(uVelocity, vL).x;
      float R = texture(uVelocity, vR).x;
      float T = texture(uVelocity, vT).y;
      float B = texture(uVelocity, vB).y;
      float div = 0.5 * (R - L + T - B);
      fragColor = vec4(div, 0.0, 0.0, 1.0);
    }`;

  const pressureSrc = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uDivergence;
    out vec4 fragColor;
    void main(){
      float L = texture(uPressure, vL).x;
      float R = texture(uPressure, vR).x;
      float T = texture(uPressure, vT).x;
      float B = texture(uPressure, vB).x;
      float C = texture(uDivergence, vUv).x;
      float p = (L + R + B + T - C) * 0.25;
      fragColor = vec4(p, 0.0, 0.0, 1.0);
    }`;

  const gradSubSrc = `#version 300 es
    precision mediump float;
    in vec2 vUv;
    in vec2 vL; in vec2 vR; in vec2 vT; in vec2 vB;
    uniform sampler2D uPressure;
    uniform sampler2D uVelocity;
    out vec4 fragColor;
    void main(){
      float L = texture(uPressure, vL).x;
      float R = texture(uPressure, vR).x;
      float T = texture(uPressure, vT).x;
      float B = texture(uPressure, vB).x;
      vec2 velocity = texture(uVelocity, vUv).xy;
      velocity.xy -= vec2(R - L, T - B);
      fragColor = vec4(velocity, 0.0, 1.0);
    }`;

  // --- Programs ---
  let clearProgram, displayProgram, splatProgram, advectionProgram;
  let curlProgram, vorticityProgram, divergenceProgram, pressureProgram, gradSubProgram;

  function initPrograms() {
    clearProgram = new Program(baseVertexSrc, clearSrc);
    displayProgram = new Program(baseVertexSrc, displaySrc);
    splatProgram = new Program(baseVertexSrc, splatSrc);
    advectionProgram = new Program(baseVertexSrc, advectionSrc);
    curlProgram = new Program(baseVertexSrc, curlSrc);
    vorticityProgram = new Program(baseVertexSrc, vorticitySrc);
    divergenceProgram = new Program(baseVertexSrc, divergenceSrc);
    pressureProgram = new Program(baseVertexSrc, pressureSrc);
    gradSubProgram = new Program(baseVertexSrc, gradSubSrc);
  }

  // --- Quad ---
  let quadVAO;

  function initQuad() {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, -1,1, 1,1, 1,-1]), gl.STATIC_DRAW);
    const idx = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0,1,2, 0,2,3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    quadVAO = vao;
  }

  function blit(target) {
    gl.bindVertexArray(quadVAO);
    if (target == null) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  }

  // --- Init FBOs ---
  function initFramebuffers() {
    const simRes = params.simResolution;
    const dyeRes = params.dyeResolution;
    const simS = getResolution(simRes);
    const dyeS = getResolution(dyeRes);
    simWidth = simS.width; simHeight = simS.height;
    dyeWidth = dyeS.width; dyeHeight = dyeS.height;

    const texType = ext.halfFloatTexType;
    const rgba = ext.formatRGBA;
    const rg = ext.formatRG;
    const r = ext.formatR;
    const filtering = ext.supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

    velocity = createDoubleFBO(simWidth, simHeight, rg.internalFormat, rg.format, texType, filtering);
    density = createDoubleFBO(dyeWidth, dyeHeight, rgba.internalFormat, rgba.format, texType, filtering);
    pressure = createDoubleFBO(simWidth, simHeight, r.internalFormat, r.format, texType, filtering);
    divergence = new FBO(simWidth, simHeight, r.internalFormat, r.format, texType, gl.NEAREST);
    curl = new FBO(simWidth, simHeight, r.internalFormat, r.format, texType, gl.NEAREST);
  }

  function getResolution(resolution) {
    let aspectRatio = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspectRatio < 1) aspectRatio = 1.0 / aspectRatio;
    const min = Math.round(resolution);
    const max = Math.round(resolution * aspectRatio);
    if (gl.drawingBufferWidth > gl.drawingBufferHeight)
      return { width: max, height: min };
    else
      return { width: min, height: max };
  }

  // --- Simulation Step ---
  function step(dt) {
    gl.disable(gl.BLEND);
    // Curl
    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.texture, 0);
    blit(curl);
    // Vorticity
    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.texture, 0);
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.texture, 1);
    gl.uniform1f(vorticityProgram.uniforms.curl, params.curl);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();
    // Divergence
    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.texture, 0);
    blit(divergence);
    // Clear pressure
    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.texture, 0);
    gl.uniform1f(clearProgram.uniforms.value, params.pressure);
    gl.uniform2f(clearProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    blit(pressure.write);
    pressure.swap();
    // Pressure
    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.texture, 0);
    for (let i = 0; i < params.pressureIterations; i++) {
      gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.texture, 1);
      blit(pressure.write);
      pressure.swap();
    }
    // Gradient subtract
    gradSubProgram.bind();
    gl.uniform2f(gradSubProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(gradSubProgram.uniforms.uPressure, pressure.read.texture, 0);
    gl.uniform1i(gradSubProgram.uniforms.uVelocity, velocity.read.texture, 1);
    blit(velocity.write);
    velocity.swap();
    // Advect velocity
    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, 1.0 / simWidth, 1.0 / simHeight);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.texture, 0);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocity.read.texture, 0);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, params.velocityDissipation);
    blit(velocity.write);
    velocity.swap();
    // Advect density
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.texture, 0);
    gl.uniform1i(advectionProgram.uniforms.uSource, density.read.texture, 1);
    gl.uniform1f(advectionProgram.uniforms.dissipation, params.densityDissipation);
    blit(density.write);
    density.swap();
  }

  // --- Splat ---
  function splat(x, y, dx, dy, color) {
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.texture, 0);
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0.0);
    gl.uniform1f(splatProgram.uniforms.radius, params.splatRadius);
    blit(velocity.write);
    velocity.swap();
    gl.uniform1i(splatProgram.uniforms.uTarget, density.read.texture, 0);
    gl.uniform3f(splatProgram.uniforms.color, color.r, color.g, color.b);
    blit(density.write);
    density.swap();
  }

  function splatPointer(x, y, dx, dy, color) {
    splat(x, y, dx * params.splatForce, dy * params.splatForce, color);
  }

  // --- Render ---
  function render() {
    displayProgram.bind();
    gl.uniform2f(displayProgram.uniforms.texelSize, 1.0 / density.read.width, 1.0 / density.read.height);
    gl.uniform1i(displayProgram.uniforms.uTexture, density.read.texture, 0);
    if (params.shading) {
      gl.uniform1i(displayProgram.uniforms.uShading, 1);
    }
    blit(null);
  }

  // --- Color ---
  function HSVtoRGB(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    let r, g, b;
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      case 5: r = v; g = p; b = q; break;
    }
    return { r, g, b };
  }

  function generateColor() {
    const c = HSVtoRGB(Math.random(), 1.0, 1.0);
    c.r *= 0.15;
    c.g *= 0.15;
    c.b *= 0.15;
    return c;
  }

  // --- Pointer Updates ---
  function correctPos(clientX, clientY) {
    return { x: clientX / canvas.width, y: 1.0 - clientY / canvas.height };
  }

  function updatePointerMoveData(pointer, posX, posY, color) {
    pointer.prevTexcoordX = pointer.texcoordX;
    pointer.prevTexcoordY = pointer.texcoordY;
    pointer.texcoordX = posX;
    pointer.texcoordY = posY;
    pointer.deltaX = correctDelta(posX - pointer.prevTexcoordX);
    pointer.deltaY = correctDelta(posY - pointer.prevTexcoordY);
    pointer.moved = Math.abs(pointer.deltaX) > 0 || Math.abs(pointer.deltaY) > 0;
    pointer.color = color;
  }

  function updatePointerDownData(pointer, id, posX, posY) {
    pointer.id = id;
    pointer.texcoordX = posX;
    pointer.texcoordY = posY;
    pointer.prevTexcoordX = posX;
    pointer.prevTexcoordY = posY;
    pointer.deltaX = 0;
    pointer.deltaY = 0;
    pointer.down = true;
    pointer.moved = false;
  }

  function correctDelta(d) {
    const rate = Math.max(canvas.width, canvas.height) / 500;
    return d * rate;
  }

  // --- Click Splat ---
  function clickSplat(pointer) {
    const color = generateColor();
    color.r *= 10;
    color.g *= 10;
    color.b *= 10;
    splatPointer(
      pointer.texcoordX, pointer.texcoordY,
      10 * (Math.random() - 0.5),
      30 * (Math.random() - 0.5),
      color
    );
  }

  // --- Animation Loop ---
  function startAnimationLoop() {
    if (animationStarted) return;
    animationStarted = true;
    lastTime = performance.now();
    requestAnimationFrame(loop);
  }

  function loop(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.016666);
    lastTime = now;

    // Update colors periodically
    colorTime += dt * params.colorUpdateSpeed;
    if (colorTime >= 1) {
      colorTime -= Math.floor(colorTime);
      pointers.forEach(p => { if (p.moved) p.color = generateColor(); });
    }

    // Splat pointers
    pointers.forEach(p => {
      if (p.moved) {
        p.moved = false;
        splatPointer(p.texcoordX, p.texcoordY, p.deltaX, p.deltaY, p.color);
      }
    });

    step(dt);
    render();
    requestAnimationFrame(loop);
  }

  // --- Events ---
  function initEvents() {
    // First mousemove starts animation
    document.body.addEventListener('mousemove', function onFirstMove(e) {
      const p = pointers[0];
      const pos = correctPos(e.clientX, e.clientY);
      updatePointerMoveData(p, pos.x, pos.y, generateColor());
      document.body.removeEventListener('mousemove', onFirstMove);
      startAnimationLoop();
    });

    window.addEventListener('mousemove', function (e) {
      const p = pointers[0];
      const pos = correctPos(e.clientX, e.clientY);
      updatePointerMoveData(p, pos.x, pos.y, p.color);
    });

    window.addEventListener('mousedown', function (e) {
      const p = pointers[0];
      const pos = correctPos(e.clientX, e.clientY);
      updatePointerDownData(p, -1, pos.x, pos.y);
      clickSplat(p);
    });

    window.addEventListener('touchstart', function (e) {
      const touches = e.targetTouches;
      for (let i = 0; i < touches.length; i++) {
        if (i >= pointers.length) pointers.push(pointer());
        const pos = correctPos(touches[i].clientX, touches[i].clientY);
        updatePointerDownData(pointers[i], touches[i].identifier, pos.x, pos.y);
        pointers[i].color = generateColor();
        clickSplat(pointers[i]);
      }
    }, { passive: true });

    window.addEventListener('touchmove', function (e) {
      const touches = e.targetTouches;
      for (let i = 0; i < touches.length; i++) {
        const pos = correctPos(touches[i].clientX, touches[i].clientY);
        updatePointerMoveData(pointers[i], pos.x, pos.y, pointers[i].color);
      }
    }, { passive: true });

    window.addEventListener('touchend', function (e) {
      const touches = e.changedTouches;
      for (let i = 0; i < touches.length; i++) {
        pointers[i].down = false;
      }
    });

    // Resize
    window.addEventListener('resize', function () {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initFramebuffers();
    });
  }

  // --- Init ---
  function init() {
    const ctx = getWebGLContext();
    if (!ctx) return;
    gl = ctx.gl;
    ext = ctx.ext;

    initPrograms();
    initQuad();
    initFramebuffers();
    initEvents();
  }

  // Wait for DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
