import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

export const BiomedicalNeuralCore: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [hasWebGL, setHasWebGL] = useState<boolean>(true);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState<boolean>(false);

  useEffect(() => {
    // Check for reduced motion
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);
    const handleMotionChange = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mediaQuery.addEventListener('change', handleMotionChange);

    // Check for WebGL capability
    try {
      const canvas = document.createElement('canvas');
      const gl = !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
      if (!gl) {
        setHasWebGL(false);
        return;
      }
    } catch {
      setHasWebGL(false);
      return;
    }

    if (mediaQuery.matches) {
      return;
    }

    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 360;
    const height = container.clientHeight || 360;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.z = 85;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // 1. Generate Biomedical Network Nodes
    const nodeCount = 38;
    const nodeGeometry = new THREE.BufferGeometry();
    const nodePositions = new Float32Array(nodeCount * 3);
    const nodeColors = new Float32Array(nodeCount * 3);

    // Color palette: Cyan, Indigo, Blue, Emerald
    const palette = [
      new THREE.Color(0x0ea5e9), // Sky blue
      new THREE.Color(0x6366f1), // Indigo
      new THREE.Color(0x14b8a6), // Teal
      new THREE.Color(0x38bdf8), // Light Cyan
    ];

    const rawNodes: THREE.Vector3[] = [];
    const radius = 24;

    for (let i = 0; i < nodeCount; i++) {
      // Golden spiral sphere distribution
      const phi = Math.acos(-1 + (2 * i) / nodeCount);
      const theta = Math.sqrt(nodeCount * Math.PI) * phi;

      const r = radius * (0.8 + Math.random() * 0.4);
      const x = r * Math.cos(theta) * Math.sin(phi);
      const y = r * Math.sin(theta) * Math.sin(phi);
      const z = r * Math.cos(phi);

      const vec = new THREE.Vector3(x, y, z);
      rawNodes.push(vec);

      nodePositions[i * 3] = x;
      nodePositions[i * 3 + 1] = y;
      nodePositions[i * 3 + 2] = z;

      const c = palette[i % palette.length];
      nodeColors[i * 3] = c.r;
      nodeColors[i * 3 + 1] = c.g;
      nodeColors[i * 3 + 2] = c.b;
    }

    nodeGeometry.setAttribute('position', new THREE.BufferAttribute(nodePositions, 3));
    nodeGeometry.setAttribute('color', new THREE.BufferAttribute(nodeColors, 3));

    // Circle texture for smooth round points
    const canvasTexture = document.createElement('canvas');
    canvasTexture.width = 32;
    canvasTexture.height = 32;
    const ctx = canvasTexture.getContext('2d');
    if (ctx) {
      ctx.beginPath();
      ctx.arc(16, 16, 14, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }
    const pointTexture = new THREE.CanvasTexture(canvasTexture);

    const nodeMaterial = new THREE.PointsMaterial({
      size: 4.5,
      vertexColors: true,
      map: pointTexture,
      transparent: true,
      opacity: 0.9,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });

    const nodesMesh = new THREE.Points(nodeGeometry, nodeMaterial);
    scene.add(nodesMesh);

    // 2. Connecting Edge Filaments
    const edgePositions: number[] = [];
    const edgeColors: number[] = [];
    const maxConnectionDist = 18;

    for (let i = 0; i < nodeCount; i++) {
      for (let j = i + 1; j < nodeCount; j++) {
        const dist = rawNodes[i].distanceTo(rawNodes[j]);
        if (dist < maxConnectionDist) {
          edgePositions.push(rawNodes[i].x, rawNodes[i].y, rawNodes[i].z);
          edgePositions.push(rawNodes[j].x, rawNodes[j].y, rawNodes[j].z);

          const c1 = palette[i % palette.length];
          const c2 = palette[j % palette.length];
          edgeColors.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b);
        }
      }
    }

    const edgeGeometry = new THREE.BufferGeometry();
    edgeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
    edgeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(edgeColors, 3));

    const edgeMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.35,
      blending: THREE.NormalBlending,
    });

    const edgeMesh = new THREE.LineSegments(edgeGeometry, edgeMaterial);
    scene.add(edgeMesh);

    // 3. Subtle ambient dust cloud
    const dustCount = 60;
    const dustGeometry = new THREE.BufferGeometry();
    const dustPositions = new Float32Array(dustCount * 3);
    for (let i = 0; i < dustCount * 3; i++) {
      dustPositions[i] = (Math.random() - 0.5) * 80;
    }
    dustGeometry.setAttribute('position', new THREE.BufferAttribute(dustPositions, 3));
    const dustMaterial = new THREE.PointsMaterial({
      color: 0x94a3b8,
      size: 1.5,
      transparent: true,
      opacity: 0.4,
    });
    const dustMesh = new THREE.Points(dustGeometry, dustMaterial);
    scene.add(dustMesh);

    // Group for gentle orbital rotation
    const coreGroup = new THREE.Group();
    coreGroup.add(nodesMesh);
    coreGroup.add(edgeMesh);
    coreGroup.add(dustMesh);
    scene.add(coreGroup);

    // Mouse interaction parallax
    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      targetX = x * 0.4;
      targetY = y * 0.4;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Animation Loop
    let animationFrameId: number;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // Slow elegant orbital rotation
      coreGroup.rotation.y += 0.0035;
      coreGroup.rotation.x += 0.0015;

      // Smooth parallax damping
      mouseX += (targetX - mouseX) * 0.05;
      mouseY += (targetY - mouseY) * 0.05;

      camera.position.x = mouseX * 25;
      camera.position.y = -mouseY * 25;
      camera.lookAt(0, 0, 0);

      renderer.render(scene, camera);
    };
    animate();

    // Resize Observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newW = entry.contentRect.width;
        const newH = entry.contentRect.height;
        if (newW > 0 && newH > 0) {
          camera.aspect = newW / newH;
          camera.updateProjectionMatrix();
          renderer.setSize(newW, newH);
        }
      }
    });
    resizeObserver.observe(container);

    // Cleanup
    return () => {
      mediaQuery.removeEventListener('change', handleMotionChange);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
      resizeObserver.disconnect();
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
      nodeGeometry.dispose();
      nodeMaterial.dispose();
      edgeGeometry.dispose();
      edgeMaterial.dispose();
      dustGeometry.dispose();
      dustMaterial.dispose();
      pointTexture.dispose();
    };
  }, [hasWebGL, prefersReducedMotion]);

  // Graceful CSS/SVG Fallback for Reduced-Motion or Non-WebGL
  if (!hasWebGL || prefersReducedMotion) {
    return (
      <div className="relative w-full h-[320px] sm:h-[380px] flex items-center justify-center">
        <div className="w-56 h-56 rounded-full border border-indigo-200/60 bg-gradient-to-tr from-indigo-50/50 via-sky-50/30 to-teal-50/50 flex items-center justify-center p-6 shadow-inner relative">
          <div className="absolute inset-4 rounded-full border border-dashed border-sky-300 animate-spin-slow" />
          <div className="text-center z-10">
            <div className="w-12 h-12 mx-auto rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-md mb-2">
              <span className="text-xl">🧬</span>
            </div>
            <div className="text-xs font-bold text-slate-800 uppercase tracking-widest font-mono">
              Biomedical Core
            </div>
            <div className="text-[11px] text-slate-500 font-medium">Multi-Model Ensemble</div>
          </div>
        </div>

        {/* Pipeline Step Tag */}
        <div className="absolute bottom-2 right-4 bg-white/90 backdrop-blur-xs border border-slate-200 px-3 py-1.5 rounded-lg shadow-xs text-[10px] font-mono font-bold text-slate-600 flex items-center space-x-1.5">
          <span className="text-indigo-600">TEXT</span>
          <span className="text-slate-400">→</span>
          <span className="text-sky-600">MODELS</span>
          <span className="text-slate-400">→</span>
          <span className="text-emerald-600">ENTITIES</span>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[320px] sm:h-[380px] select-none flex items-center justify-center">
      {/* 3D Canvas Mounting Node */}
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Decorative Outer Aura */}
      <div className="absolute inset-0 pointer-events-none bg-radial from-indigo-500/5 via-sky-500/5 to-transparent rounded-full blur-2xl" />

      {/* Pipeline Step Badge Overlay */}
      <div className="absolute bottom-2 right-2 bg-white/95 backdrop-blur-md border border-slate-200/90 px-3 py-1.5 rounded-xl shadow-xs text-[11px] font-mono font-bold text-slate-700 flex items-center space-x-2 pointer-events-none">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
        <span className="text-indigo-700 font-extrabold">TEXT</span>
        <span className="text-slate-300">↓</span>
        <span className="text-sky-700 font-extrabold">MODELS</span>
        <span className="text-slate-300">↓</span>
        <span className="text-teal-700 font-extrabold">ENTITIES</span>
      </div>
    </div>
  );
};
