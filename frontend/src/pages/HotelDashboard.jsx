import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { jsPDF } from "jspdf";
import autoTable from "jspdf-autotable";

/** Full base including `/api`. Override with REACT_APP_API_URL e.g. http://127.0.0.1:5001/api if port 5000 is taken. */
const API = (process.env.REACT_APP_API_URL || "http://127.0.0.1:5000/api").replace(/\/$/, "");

/** Fixed window for API calls (time-range UI removed). */
const TIMEFRAME = "all";

async function fetchJsonApi(url) {
  const res = await fetch(url);
  const text = await res.text();
  const ct = (res.headers.get("content-type") || "").toLowerCase();
  if (!ct.includes("application/json")) {
    const htmlHint = text.trim().toLowerCase().startsWith("<!doctype")
      ? " The server sent HTML (often the React dev server on the wrong port, or Flask not running). Run Flask from the `api` folder: python app.py — and use a different frontend port if 5000 is already in use."
      : "";
    throw new Error(
      `Expected JSON from API (HTTP ${res.status}, ${ct || "unknown content-type"}).${htmlHint}`
    );
  }
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("API returned invalid JSON.");
  }
}

/* ── Maison Velour–inspired theme: warm charcoal, cream, gold, forest accents ── */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,500;0,600;1,400&display=swap');
:root {
  --bg:#1a1714;
  --surface:#221f1c;
  --surface2:#2a2622;
  --border:rgba(232,228,219,0.09);
  --text:#e8e4db;
  --muted:#a09d98;
  --gold:#c5a059;
  --gold-dim:rgba(197,160,89,0.14);
  --gold-line:rgba(197,160,89,0.45);
  --forest:#2d3e2d;
  --forest-text:#c8d4c8;
  --red:#c45a5a;
  --amber:#c9a227;
  --purple:#9a8b9e;
  --teal:#6d8b7a;
  --accent:var(--gold);
  --green:#4a6b4e;
  --fd:'Playfair Display',Georgia,serif;
  --fb:'Inter',system-ui,sans-serif;
  --fn:'Calibri','Calibri',Candara,'Segoe UI',sans-serif;
  --ease-smooth:cubic-bezier(0.22,1,0.36,1);
  --ease-out-expo:cubic-bezier(0.16,1,0.3,1);
  --dur-panel:.45s;
}
.hd-num{font-family:var(--fn);font-weight:700;font-variant-numeric:tabular-nums;}
.hd-root *{box-sizing:border-box;margin:0;padding:0;}
.hd-root{font-family:var(--fb);font-size:14px;color:var(--text);background:var(--bg);min-height:100vh;display:flex;position:relative;letter-spacing:0.01em;}
.hd-root::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(197,160,89,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(197,160,89,.04) 1px,transparent 1px);background-size:48px 48px;pointer-events:none;z-index:0;}
.hd-sb{position:fixed;left:0;top:0;bottom:0;width:228px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:32px 0;z-index:10;}
.hd-logo{padding:0 24px 32px;border-bottom:1px solid var(--border);}
.hd-logo-mark{display:flex;align-items:center;gap:12px;}
.hd-logo-icon{width:36px;height:36px;background:var(--gold-dim);border:1px solid var(--gold-line);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:16px;color:var(--gold);}
.hd-logo-text{font-family:var(--fd);font-size:20px;font-weight:500;letter-spacing:0.02em;color:var(--text);}
.hd-logo-sub{font-size:10px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-top:4px;padding-left:48px;}
.hd-nav{padding:24px 14px;flex:1;display:flex;flex-direction:column;gap:4px;}
.hd-nl{font-size:10px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);padding:10px 12px 8px;}
.hd-ni{display:flex;align-items:center;gap:10px;padding:11px 14px;border-radius:10px;cursor:pointer;color:var(--muted);transition:background .28s var(--ease-smooth),color .22s ease,transform .2s var(--ease-smooth),box-shadow .25s ease;font-size:13.5px;font-weight:400;}
.hd-ni:hover{background:var(--surface2);color:var(--text);}
.hd-ni:active{transform:scale(0.98);}
.hd-ni.active{background:var(--gold-dim);color:var(--gold);font-weight:500;box-shadow:inset 0 0 0 1px rgba(197,160,89,.2);}
.hd-ni .dot{width:7px;height:7px;border-radius:50%;background:currentColor;opacity:.55;}
.hd-badge{margin-left:auto;background:var(--forest);color:var(--forest-text);font-family:var(--fn);font-size:10px;font-weight:700;padding:2px 8px;border-radius:99px;letter-spacing:.04em;}
.hd-main{margin-left:228px;min-height:100vh;position:relative;z-index:1;flex:1;}
.hd-tb{position:sticky;top:0;z-index:9;display:flex;align-items:center;justify-content:space-between;padding:22px 40px;background:rgba(26,23,20,.92);backdrop-filter:blur(14px);border-bottom:1px solid var(--border);}
.hd-tb-title{font-family:var(--fd);font-size:28px;font-weight:500;letter-spacing:0.02em;color:var(--text);transition:opacity .35s var(--ease-smooth);}
.hd-tb-sub{font-size:13px;color:var(--muted);margin-top:6px;font-weight:400;}
.hd-tb-right{display:flex;align-items:center;gap:12px;}
.hd-live{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:500;color:var(--forest-text);background:rgba(45,62,45,.35);border:1px solid rgba(74,107,78,.45);padding:6px 12px;border-radius:99px;letter-spacing:.06em;text-transform:uppercase;}
.hd-live-dot{width:6px;height:6px;border-radius:50%;background:#6d9072;animation:hd-blink 1.4s ease-in-out infinite;}
@keyframes hd-blink{0%,100%{opacity:1}50%{opacity:.25}}
.hd-btn{display:flex;align-items:center;gap:6px;padding:9px 16px;border-radius:8px;font-family:var(--fb);font-size:12.5px;font-weight:500;cursor:pointer;transition:background .3s var(--ease-smooth),border-color .3s ease,color .2s ease,transform .2s var(--ease-smooth),box-shadow .35s ease;}
.hd-btn-p{background:transparent;color:var(--gold);border:1px solid var(--gold-line);}
.hd-btn-p:hover{background:var(--gold-dim);border-color:var(--gold);box-shadow:0 4px 20px rgba(197,160,89,.12);}
.hd-btn-g{background:transparent;color:var(--text);border:1px solid var(--border);}
.hd-btn-g:hover{background:var(--surface2);border-color:rgba(197,160,89,.25);}
.hd-btn:active:not(:disabled){transform:scale(0.98);}
.hd-btn:disabled{opacity:.45;cursor:not-allowed;}
.hd-content{padding:36px 40px 56px;}
.hd-alert{display:flex;align-items:center;gap:14px;background:rgba(196,90,90,.08);border:1px solid rgba(196,90,90,.22);border-radius:12px;padding:14px 18px;margin-bottom:24px;animation:hd-alert-in .5s var(--ease-out-expo) both;}
@keyframes hd-alert-in{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
.hd-alert-x{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:6px 8px;line-height:1;border-radius:6px;transition:background .2s ease,color .2s ease;}
.hd-alert-x:hover{background:rgba(255,255,255,.06);color:var(--text);}
.hd-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:28px;}
.hd-mc{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:24px 26px 24px 23px;position:relative;overflow:hidden;border-left:3px solid transparent;transition:border-color var(--dur-panel) var(--ease-smooth),box-shadow var(--dur-panel) var(--ease-smooth),transform .4s var(--ease-smooth);animation:hd-up .55s var(--ease-out-expo) both;box-shadow:0 8px 32px rgba(0,0,0,.12);}
.hd-mc.c-blue{border-left-color:rgba(197,160,89,.65);}
.hd-mc.c-green{border-left-color:rgba(111,167,120,.55);}
.hd-mc.c-red{border-left-color:rgba(196,90,90,.5);}
.hd-mc:hover{border-color:rgba(197,160,89,.18);box-shadow:0 16px 44px rgba(0,0,0,.2);transform:translateY(-2px);}
.hd-mc::after{content:'';position:absolute;top:0;right:0;width:100px;height:100px;border-radius:50%;opacity:.07;transform:translate(28px,-28px);}
.hd-mc.c-blue::after{background:var(--gold);}
.hd-mc.c-green::after{background:var(--green);}
.hd-mc.c-red::after{background:var(--red);}
.hd-mc.c-amber::after{background:var(--amber);}
.hd-mc-top{display:flex;align-items:center;justify-content:flex-end;margin-bottom:14px;min-height:26px;}
.hd-trend{font-family:var(--fn);font-size:11px;font-weight:700;padding:4px 10px;border-radius:99px;transition:transform .25s var(--ease-smooth),opacity .2s ease;}
.hd-trend.up{background:rgba(74,107,78,.2);color:#9dc4a4;}
.hd-trend.down{background:rgba(196,90,90,.12);color:#e09090;}
.hd-trend.neu{background:rgba(160,157,152,.12);color:var(--muted);}
.hd-mv{font-family:var(--fn);font-size:34px;font-weight:700;line-height:1.1;margin-bottom:6px;color:var(--text);}
.hd-ml{font-size:13px;color:var(--muted);font-weight:400;}
.hd-row2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;}
.hd-panel{background:var(--surface);border:1px solid var(--border);border-radius:16px;overflow:hidden;animation:hd-up .55s var(--ease-out-expo) both;box-shadow:0 8px 32px rgba(0,0,0,.1);transition:border-color .4s var(--ease-smooth),box-shadow .5s var(--ease-smooth);}
.hd-panel:hover{border-color:rgba(197,160,89,.1);box-shadow:0 14px 42px rgba(0,0,0,.14);}
.hd-ph{padding:22px 26px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.hd-pt{font-size:15px;font-weight:500;font-family:var(--fd);display:flex;align-items:center;flex-wrap:wrap;gap:6px 12px;color:var(--text);line-height:1.4;flex:1;min-width:0;}
.hd-ph-tags{display:flex;align-items:center;gap:8px;flex-shrink:0;}
.hd-pb{padding:22px 26px;}
.hd-quote{transition:border-color .3s ease,box-shadow .35s var(--ease-smooth);}
.hd-quote:hover{border-color:rgba(197,160,89,.12)!important;box-shadow:0 4px 20px rgba(0,0,0,.08);}
.hd-cr{display:flex;align-items:center;gap:14px;margin-bottom:16px;}
.hd-cr:last-child{margin-bottom:0;}
.hd-cr.hd-cr-go{cursor:pointer;border-radius:10px;padding:8px 10px;margin:-8px -10px 10px -10px;transition:background .28s var(--ease-smooth),box-shadow .25s ease;}
.hd-cr.hd-cr-go:hover{background:var(--surface2);}
.hd-cr.hd-cr-go.on{box-shadow:inset 0 0 0 1px var(--gold-line);}
.hd-cl{width:52px;font-size:12px;font-weight:500;color:var(--gold);letter-spacing:.04em;text-transform:uppercase;}
.hd-bt{flex:1;height:7px;background:var(--surface2);border-radius:99px;overflow:hidden;}
.hd-bf{height:100%;border-radius:99px;transform-origin:left;animation:hd-bar .85s var(--ease-out-expo) both;transition:width .65s var(--ease-smooth);}
@keyframes hd-bar{from{transform:scaleX(0)}to{transform:scaleX(1)}}
.hd-bf.rooms{background:linear-gradient(90deg,#c5a059,#d4b87a);animation-delay:.25s;}
.hd-bf.staff{background:linear-gradient(90deg,#8b7d9e,#a898ad);animation-delay:.32s;}
.hd-bf.food{background:linear-gradient(90deg,#c9a227,#dfc66a);animation-delay:.39s;}
.hd-bf.other{background:linear-gradient(90deg,#5d7a68,#8faa96);animation-delay:.46s;}
.hd-cc{width:36px;text-align:right;font-family:var(--fn);font-size:14px;font-weight:700;}
.hd-cp{width:40px;text-align:right;font-family:var(--fn);font-size:11px;font-weight:700;color:var(--muted);}
.hd-cat-block{margin-bottom:20px;}
.hd-cat-block:last-child{margin-bottom:0;}
.hd-actions-box{margin-top:12px;margin-left:66px;margin-right:0;padding:14px 16px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;border-left:3px solid var(--gold);}
.hd-actions-title{font-size:10px;text-transform:uppercase;letter-spacing:.12em;font-weight:600;color:var(--gold);margin-bottom:10px;}
.hd-actions-ul{margin:0;padding-left:18px;font-size:13px;line-height:1.6;color:var(--muted);}
.hd-actions-ul li{margin-bottom:6px;}
.hd-actions-empty{font-size:12px;color:var(--muted);font-style:italic;}
.hd-actions-err{font-size:12px;color:var(--red);}
.hd-sr{display:flex;align-items:center;gap:12px;margin-bottom:14px;}
.hd-spill{display:flex;align-items:center;gap:6px;padding:6px 14px;border-radius:99px;font-size:11px;font-weight:600;min-width:92px;letter-spacing:.06em;text-transform:uppercase;}
.hd-spill.high{background:rgba(196,90,90,.12);color:#e8a0a0;border:1px solid rgba(196,90,90,.25);}
.hd-spill.medium{background:rgba(201,162,39,.12);color:#dfc06a;border:1px solid rgba(201,162,39,.22);}
.hd-spill.low{background:rgba(74,107,78,.22);color:#9dc4a4;border:1px solid rgba(74,107,78,.3);}
.hd-sbt{flex:1;height:6px;background:var(--surface2);border-radius:99px;overflow:hidden;}
.hd-sb2{height:100%;border-radius:99px;transition:width .6s ease;}
.hd-sb2.high{background:linear-gradient(90deg,#a05050,#c45a5a);}
.hd-sb2.medium{background:linear-gradient(90deg,#a88620,#c9a227);}
.hd-sb2.low{background:linear-gradient(90deg,#4a6b4e,#6d9072);}
.hd-sn{font-family:var(--fn);font-size:13px;font-weight:700;width:28px;text-align:right;}
.hd-dw{display:flex;align-items:center;gap:28px;}
.hd-dl{display:flex;flex-direction:column;gap:12px;}
.hd-dli{display:flex;align-items:center;gap:10px;font-size:13px;}
.hd-dld{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.hd-dlv{margin-left:auto;font-family:var(--fn);font-weight:700;font-size:14px;}
.hd-tbl{width:100%;border-collapse:collapse;}
.hd-tbl th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);padding:0 14px 14px;border-bottom:1px solid var(--border);font-weight:600;}
.hd-tbl td{padding:16px 14px;border-bottom:1px solid rgba(232,228,219,.06);vertical-align:top;}
.hd-tbl tr:last-child td{border-bottom:none;}
.hd-tbl tbody tr{transition:background .28s var(--ease-smooth),box-shadow .25s ease;cursor:pointer;}
.hd-tbl tbody tr:hover td{background:rgba(197,160,89,.06);}
.hd-tbl tbody tr.selected td{background:rgba(197,160,89,.11);}
.hd-tbl tbody tr.selected td:first-child{box-shadow:inset 3px 0 0 var(--gold);}
.hd-rv{font-size:13px;line-height:1.55;max-width:280px;color:var(--muted);display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.hd-cb{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:600;white-space:nowrap;letter-spacing:.04em;text-transform:uppercase;}
.hd-cb.Rooms{background:rgba(197,160,89,.15);color:#e0c88a;border:1px solid rgba(197,160,89,.25);}
.hd-cb.Staff{background:rgba(154,139,158,.15);color:#c4b5c9;border:1px solid rgba(154,139,158,.22);}
.hd-cb.Food{background:rgba(201,162,39,.14);color:#dfc06a;border:1px solid rgba(201,162,39,.22);}
.hd-cb.Other{background:rgba(109,139,122,.16);color:#a0baa8;border:1px solid rgba(109,139,122,.25);}
.hd-cb.Positive{background:rgba(74,107,78,.22);color:#9dc4a4;border:1px solid rgba(74,107,78,.35);}
.hd-cb.Negative{background:rgba(196,90,90,.12);color:#e8a0a0;border:1px solid rgba(196,90,90,.22);}
.hd-st{display:inline-block;padding:4px 10px;border-radius:6px;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;}
.hd-st.High{background:rgba(196,90,90,.14);color:#e8a0a0;}
.hd-st.Medium{background:rgba(201,162,39,.14);color:#dfc06a;}
.hd-st.Low{background:rgba(74,107,78,.2);color:#9dc4a4;}
.hd-st.dash{background:rgba(120,120,130,.12);color:var(--muted);text-transform:none;letter-spacing:0;}
.hd-prev{font-size:12px;color:var(--muted);line-height:1.5;max-width:220px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.hd-sd{display:inline-flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);}
.hd-sd::before{content:'';width:6px;height:6px;border-radius:50%;display:inline-block;}
.hd-sd.pending::before{background:var(--amber);}
.hd-sd.resolved::before{background:#6d9072;}
.hd-tabs{display:flex;gap:3px;background:var(--surface2);border-radius:10px;padding:4px;border:1px solid var(--border);}
.hd-tab{padding:8px 16px;border-radius:8px;font-size:12.5px;cursor:pointer;color:var(--muted);transition:background .3s var(--ease-smooth),color .22s ease,transform .2s ease,box-shadow .25s ease;border:none;background:none;font-family:var(--fb);font-weight:400;}
.hd-tab:hover{color:var(--text);}
.hd-tab.active{background:var(--surface);color:var(--text);font-weight:500;box-shadow:0 2px 12px rgba(0,0,0,.18);}
.hd-sol-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.hd-sc{background:var(--surface2);border:1px solid var(--border);border-radius:14px;padding:18px 18px 18px 16px;border-left:3px solid rgba(197,160,89,.35);transition:border-color .35s var(--ease-smooth),box-shadow .4s var(--ease-smooth),transform .4s var(--ease-smooth);animation:hd-up-card .55s var(--ease-out-expo) both;}
.hd-sc:hover{border-color:rgba(197,160,89,.2);box-shadow:0 10px 28px rgba(0,0,0,.12);transform:translateY(-2px);}
.hd-sc--urgent{border-left-color:rgba(196,90,90,.55);}
.hd-sc--week{border-left-color:rgba(201,162,39,.5);}
.hd-sc--month{border-left-color:rgba(197,160,89,.6);}
.hd-sc--reply{border-left-color:rgba(74,107,78,.5);}
.hd-sol-grid .hd-sc{margin-bottom:0;}
@keyframes hd-up-card{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.hd-sl{font-size:10px;text-transform:uppercase;letter-spacing:.12em;font-weight:600;color:var(--gold);margin-bottom:10px;line-height:1.35;}
.hd-stx{font-size:13px;color:var(--muted);line-height:1.65;}
.hd-chip{flex:1;background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:12px 16px;font-size:12px;transition:border-color .3s var(--ease-smooth),box-shadow .35s ease,transform .3s var(--ease-smooth);}
.hd-chip:hover{border-color:rgba(197,160,89,.15);transform:translateY(-1px);}
.hd-chip-l{color:var(--muted);margin-bottom:4px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;}
.hd-chip-v{font-weight:500;color:var(--text);}
.hd-skeleton{background:linear-gradient(90deg,var(--surface2) 25%,#2f2b27 50%,var(--surface2) 75%);background-size:200% 100%;animation:hd-shimmer 1.5s infinite;border-radius:8px;}
@keyframes hd-shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
.hd-loading-row{height:22px;margin-bottom:10px;}
.hd-spinner{width:16px;height:16px;border:2px solid var(--border);border-top-color:var(--gold);border-radius:50%;animation:hd-spin .7s linear infinite;display:inline-block;}
@keyframes hd-spin{to{transform:rotate(360deg)}}
.hd-err{color:var(--red);font-size:12px;padding:8px 0;}
.hd-toast-host{position:fixed;bottom:24px;right:24px;z-index:10000;display:flex;flex-direction:column;gap:10px;pointer-events:none;max-width:min(440px,calc(100vw - 40px));}
.hd-toast{pointer-events:auto;display:flex;align-items:flex-start;gap:12px;padding:14px 16px;border-radius:12px;border:1px solid var(--border);background:var(--surface);box-shadow:0 14px 48px rgba(0,0,0,.4);animation:hd-toast-in .38s var(--ease-out-expo) both;font-size:13px;line-height:1.5;color:var(--text);}
@keyframes hd-toast-in{from{opacity:0;transform:translateY(12px) scale(.98)}to{opacity:1;transform:translateY(0) scale(1)}}
.hd-toast.info{border-color:rgba(197,160,89,.28);background:rgba(197,160,89,.08);}
.hd-toast.success{border-color:rgba(74,107,78,.45);background:rgba(45,62,45,.35);}
.hd-toast.error{border-color:rgba(196,90,90,.4);background:rgba(196,90,90,.12);}
.hd-toast.warning{border-color:rgba(201,162,39,.4);background:rgba(201,162,39,.1);}
.hd-toast-x{flex-shrink:0;background:none;border:none;color:var(--muted);cursor:pointer;font-size:18px;line-height:1;padding:2px 6px;border-radius:6px;transition:background .2s ease,color .2s ease;}
.hd-toast-x:hover{background:rgba(255,255,255,.08);color:var(--text);}
.hd-modal-backdrop{position:fixed;inset:0;background:rgba(10,8,6,.65);z-index:10001;display:flex;align-items:center;justify-content:center;padding:24px;backdrop-filter:blur(6px);}
.hd-modal{max-width:440px;width:100%;background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:26px 26px 22px;box-shadow:0 28px 90px rgba(0,0,0,.55);}
.hd-modal-title{font-family:var(--fd);font-size:19px;font-weight:500;color:var(--text);margin-bottom:12px;}
.hd-modal-body{color:var(--muted);font-size:13px;line-height:1.6;margin-bottom:22px;white-space:pre-wrap;}
.hd-modal-actions{display:flex;justify-content:flex-end;gap:10px;flex-wrap:wrap;}
.hd-btn-danger{background:transparent;color:var(--red);border:1px solid rgba(196,90,90,.45);}
.hd-btn-danger:hover{background:rgba(196,90,90,.12);border-color:var(--red);}
@keyframes hd-up{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.hd-empty-hint{animation:hd-fade-in .6s var(--ease-out-expo) .08s both;}
@keyframes hd-fade-in{from{opacity:0}to{opacity:1}}
.hd-metrics .hd-mc:nth-child(1){animation-delay:.05s;}
.hd-metrics .hd-mc:nth-child(2){animation-delay:.1s;}
.hd-metrics .hd-mc:nth-child(3){animation-delay:.15s;}
.hd-sol-grid .hd-sc:nth-child(1){animation-delay:.06s;}
.hd-sol-grid .hd-sc:nth-child(2){animation-delay:.1s;}
.hd-sol-grid .hd-sc:nth-child(3){animation-delay:.14s;}
.hd-sol-grid .hd-sc:nth-child(4){animation-delay:.18s;}
`;

/* ── small helpers ── */
const CAT_COLORS = {
  Rooms: "#e0c88a",
  Staff: "#c4b5c9",
  Food: "#dfc06a",
  Other: "#a0baa8",
};
const CAT_CLS = {
  Rooms: "rooms", Staff: "staff", Food: "food", Other: "other"
};

function Skeleton({ w = "100%", h = 18 }) {
  return <div className="hd-skeleton" style={{ width: w, height: h }} />;
}

function MetricCard({ label, value, trend, trendDir, color, loading }) {
  return (
    <div className={`hd-mc c-${color}`}>
      <div className="hd-mc-top">
        {loading
          ? <Skeleton w={72} h={22} />
          : <span className={`hd-trend ${trendDir}`}>{trend}</span>}
      </div>
      {loading
        ? <><Skeleton w="60%" h={32} /><div style={{ marginTop: 6 }} /><Skeleton w="80%" h={14} /></>
        : <><div className="hd-mv">{value}</div><div className="hd-ml">{label}</div></>
      }
    </div>
  );
}

function SolCard({ variant, label, text }) {
  return (
    <div className={`hd-sc hd-sc--${variant}`}>
      <div className="hd-sl">{label}</div>
      <div className="hd-stx">{text}</div>
    </div>
  );
}

/* ── MAIN COMPONENT ── */
export default function HotelDashboard() {
  const [summary, setSummary] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [selectedRow, setSelectedRow] = useState(null);
  const [solution, setSolution] = useState(null);
  const [solLoading, setSolLoading] = useState(false);
  const [solError, setSolError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);
  const [activeTab, setActiveTab] = useState("All");
  const [activeNav, setActiveNav] = useState(0);
  const [showAlert, setShowAlert] = useState(true);
  const [activeCategory, setActiveCategory] = useState("All");
  const [uploading, setUploading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [reviewsLoading, setReviewsLoading] = useState(true);
  const [reviewsTotalMatching, setReviewsTotalMatching] = useState(0);
  const [reviewsRefreshToken, setReviewsRefreshToken] = useState(0);
  const [categoryActions, setCategoryActions] = useState(null);
  const [categoryActionsLoading, setCategoryActionsLoading] = useState(false);
  const [categoryActionsError, setCategoryActionsError] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [confirmDialog, setConfirmDialog] = useState(null);
  const fileInputRef = useRef(null);
  const toastIdRef = useRef(0);

  const pushToast = useCallback((message, variant = "info") => {
    const id = ++toastIdRef.current;
    const duration = variant === "error" ? 9000 : variant === "warning" ? 7500 : 5600;
    setToasts((prev) => [...prev, { id, message, variant }]);
    window.setTimeout(() => {
      setToasts((prev) => prev.filter((x) => x.id !== id));
    }, duration);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((prev) => prev.filter((x) => x.id !== id));
  }, []);

  useEffect(() => {
    if (!confirmDialog) return;
    const onKey = (e) => {
      if (e.key === "Escape") setConfirmDialog(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [confirmDialog]);

  const fetchAllData = useCallback(() => {
    setLoading(true);
    setCategoryActionsLoading(true);
    setCategoryActionsError(null);
    setFetchError(null);

    Promise.all([fetchJsonApi(`${API}/summary?timeframe=${TIMEFRAME}`)])
      .then(([sum]) => {
        if (sum.error) throw new Error(sum.error);
        setSummary(sum);
        setLoading(false);
        const highCount = sum.severities?.High || 0;
        setShowAlert(highCount > 0);
      })
      .catch((err) => {
        setSummary(null);
        setFetchError(err.message);
        setLoading(false);
      });

    fetchJsonApi(`${API}/category_actions?timeframe=${TIMEFRAME}`)
      .then((catAct) => {
        if (catAct.error) {
          setCategoryActions(null);
          setCategoryActionsError(catAct.error);
        } else {
          setCategoryActions(catAct.categories || null);
        }
      })
      .catch((err) => {
        setCategoryActions(null);
        setCategoryActionsError(err.message);
      })
      .finally(() => {
        setCategoryActionsLoading(false);
      });
  }, []);

  /* ── summary + category actions ── */
  useEffect(() => {
    fetchAllData();
  }, [fetchAllData]);

  /* ── reviews: refetch when category changes (server filters category across full data) ── */
  useEffect(() => {
    let cancelled = false;
    setReviewsLoading(true);
    setSelectedRow(null);
    setSolution(null);
    const catQ =
      activeCategory !== "All"
        ? `&category=${encodeURIComponent(activeCategory)}`
        : "";
    fetchJsonApi(`${API}/reviews?timeframe=${TIMEFRAME}${catQ}`)
      .then(data => {
        if (cancelled) return;
        if (data.error) throw new Error(data.error);
        setReviews(data.reviews || []);
        setReviewsTotalMatching(
          typeof data.total_matching === "number"
            ? data.total_matching
            : (data.reviews || []).length
        );
      })
      .catch(err => {
        if (!cancelled) {
          setReviews([]);
          setReviewsTotalMatching(0);
          pushToast(err.message || "Could not load reviews.", "error");
        }
      })
      .finally(() => {
        if (!cancelled) setReviewsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeCategory, reviewsRefreshToken, pushToast]);

  /* ── fetch solution when a row is selected ── */
  const fetchSolution = useCallback(async (review) => {
    setSolution(null);
    setSolError(null);
    setSolLoading(true);
    try {
      const res = await fetch(`${API}/review/solution`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          review_text: review.review_text,
          category: review.category,
          severity: review.severity,
          tags: review.tags,
          reviewer_score:
            review.reviewer_score != null && review.reviewer_score !== ""
              ? review.reviewer_score
              : undefined,
        }),
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setSolution(data.solution);
    } catch (e) {
      setSolError(e.message || "Could not generate solution.");
    } finally {
      setSolLoading(false);
    }
  }, []);

  const handleRowClick = (review) => {
    setSelectedRow(review.id);
    fetchSolution(review);
    setActiveNav(2); // Automatically switch to Solutions tab
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/upload_monthly`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      fetchAllData();
      setReviewsRefreshToken(t => t + 1);
      const n =
        typeof data.rows_processed === "number" ? data.rows_processed : null;
      pushToast(
        n != null
          ? `Batch complete: ${n.toLocaleString()} review${n === 1 ? "" : "s"} processed. Dashboard updated.`
          : "Batch processing complete! Dashboard updated with new data.",
        "success"
      );
    } catch (err) {
      pushToast("Upload failed: " + err.message, "error");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleSyncFromMongo = async () => {
    setSyncing(true);
    try {
      const res = await fetch(`${API}/sync_from_mongo`, { method: "POST" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }
      fetchAllData();
      setReviewsRefreshToken((t) => t + 1);
      const synced = typeof data.rows_synced === "number" ? data.rows_synced : null;
      const proc = typeof data.rows_processed === "number" ? data.rows_processed : null;
      pushToast(
        synced != null && proc != null
          ? `Synced ${synced.toLocaleString()} review${synced === 1 ? "" : "s"} from MongoDB. Batch pipeline finished ${proc.toLocaleString()} row${proc === 1 ? "" : "s"}. Dashboard updated.`
          : "MongoDB sync complete. Dashboard updated.",
        "success"
      );
    } catch (err) {
      pushToast("Sync from database failed: " + err.message, "error");
    } finally {
      setSyncing(false);
    }
  };

  const handleClearAllReviews = () => {
    setConfirmDialog({
      title: "Clear all review data?",
      message:
        "This removes preprocessed and clustered CSV rows, monthly batch files, and the uploaded monthly CSV from the dashboard.\n\n" +
        "Trained ML model files (.pkl) are not deleted.\n\n" +
        "This cannot be undone.",
      confirmLabel: "Clear all data",
      onConfirm: () => {
        setConfirmDialog(null);
        void (async () => {
          setClearing(true);
          try {
            const res = await fetch(`${API}/clear_reviews`, { method: "POST" });
            const text = await res.text();
            let data;
            try {
              data = JSON.parse(text);
            } catch {
              throw new Error(text.slice(0, 160) || "Invalid response from server");
            }
            if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
            setSelectedRow(null);
            setSolution(null);
            fetchAllData();
            setReviewsRefreshToken((t) => t + 1);
            pushToast("All review data cleared. You can upload a new CSV to start fresh.", "success");
          } catch (err) {
            pushToast("Clear failed: " + err.message, "error");
          } finally {
            setClearing(false);
          }
        })();
      },
    });
  };

  const handleExportPdf = useCallback(() => {
    if (!summary) {
      pushToast("Summary is not loaded yet. Please try again in a moment.", "warning");
      return;
    }

    const now = new Date();
    const reportDate = now.toLocaleString();
    const periodLabel = "All Time";

    const total = Math.max(0, Math.trunc(Number(summary.total) || 0));
    const positive = Math.max(0, Math.trunc(Number(summary.positive) || 0));
    const negative = Math.max(0, Math.trunc(Number(summary.negative) || 0));
    const posPct = total > 0 ? ((positive / total) * 100).toFixed(1) : "0.0";
    const negPct = total > 0 ? ((negative / total) * 100).toFixed(1) : "0.0";

    const orderedCats = ["Rooms", "Staff", "Food", "Other"];
    const categoryRows = orderedCats.map((cat) => {
      const count = Math.max(0, Math.trunc(Number(summary?.categories?.[cat]) || 0));
      const pct = negative > 0 ? ((count / negative) * 100).toFixed(1) : "0.0";
      return [cat, String(count), `${pct}%`];
    });

    const actionRows = orderedCats.map((cat) => {
      const count = Math.max(0, Math.trunc(Number(summary?.categories?.[cat]) || 0));
      const actions = categoryActions?.[cat]?.actions || [];
      const actionText = actions.length > 0
        ? actions.slice(0, 4).map((a, idx) => `${idx + 1}. ${a}`).join("\n")
        : "No AI actions generated for this category in selected period.";
      return [cat, String(count), actionText];
    });

    const decisionRows = orderedCats.map((cat) => {
      const actions = categoryActions?.[cat]?.actions || [];
      if (actions.length === 0) {
        return [cat, "No decision needed now", "Monitor and review next reporting cycle."];
      }
      const decision = `Assign an owner for ${cat} and execute top action: ${actions[0]}`;
      const timeline = actions[1]
        ? `Start immediately, then follow with: ${actions[1]}`
        : "Start immediately with daily tracking until complaints decline.";
      return [cat, decision, timeline];
    });

    const doc = new jsPDF({ orientation: "p", unit: "pt", format: "a4" });
    const left = 40;
    const right = 555;
    let y = 42;

    doc.setFillColor(19, 24, 31);
    doc.rect(0, 0, 595, 92, "F");
    doc.setTextColor(232, 234, 240);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(20);
    doc.text("Hotel Review Intelligence Report", left, y);
    y += 20;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.text(`Generated: ${reportDate}`, left, y);
    doc.text(`Period: ${periodLabel}`, right - doc.getTextWidth(`Period: ${periodLabel}`), y);
    y += 28;

    doc.setTextColor(20, 24, 33);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(13);
    doc.text("Executive Summary", left, y);
    y += 16;
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10.5);
    const summaryText =
      `Dataset contains ${total.toLocaleString()} reviews with ${positive.toLocaleString()} positive ` +
      `(${posPct}%) and ${negative.toLocaleString()} negative (${negPct}%). ` +
      `The decisions below prioritize category actions for managers.`;
    doc.text(doc.splitTextToSize(summaryText, 515), left, y);
    y += 34;

    autoTable(doc, {
      startY: y,
      head: [["Metric", "Value"]],
      body: [
        ["Total reviews", total.toLocaleString()],
        ["Positive reviews", `${positive.toLocaleString()} (${posPct}%)`],
        ["Negative reviews", `${negative.toLocaleString()} (${negPct}%)`],
      ],
      styles: { fontSize: 10, cellPadding: 6, lineColor: [230, 233, 238], lineWidth: 0.4 },
      headStyles: { fillColor: [39, 86, 166], textColor: 255, fontStyle: "bold" },
      margin: { left, right: 40 },
    });

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 16,
      head: [["Category", "Negative Count", "% of Negative"]],
      body: categoryRows,
      styles: { fontSize: 10, cellPadding: 6, lineColor: [230, 233, 238], lineWidth: 0.4 },
      headStyles: { fillColor: [15, 92, 77], textColor: 255, fontStyle: "bold" },
      margin: { left, right: 40 },
    });

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 16,
      head: [["Category", "Complaints", "AI Actions to Take"]],
      body: actionRows,
      styles: { fontSize: 9.5, cellPadding: 6, valign: "top", lineColor: [230, 233, 238], lineWidth: 0.4 },
      headStyles: { fillColor: [79, 142, 247], textColor: 255, fontStyle: "bold" },
      margin: { left, right: 40 },
      columnStyles: { 0: { cellWidth: 80 }, 1: { cellWidth: 80 }, 2: { cellWidth: 355 } },
    });

    autoTable(doc, {
      startY: doc.lastAutoTable.finalY + 16,
      head: [["Category", "Manager Decision", "Execution Note"]],
      body: decisionRows,
      styles: { fontSize: 9.5, cellPadding: 6, valign: "top", lineColor: [230, 233, 238], lineWidth: 0.4 },
      headStyles: { fillColor: [124, 58, 237], textColor: 255, fontStyle: "bold" },
      margin: { left, right: 40, bottom: 40 },
      columnStyles: { 0: { cellWidth: 80 }, 1: { cellWidth: 210 }, 2: { cellWidth: 225 } },
      didDrawPage: () => {
        doc.setFontSize(9);
        doc.setTextColor(110, 118, 133);
        doc.text(
          "Confidential - For internal management planning only",
          left,
          doc.internal.pageSize.getHeight() - 18
        );
      },
    });

    const fname = `hotel_management_report_${now.toISOString().slice(0, 10)}.pdf`;
    doc.save(fname);
    pushToast(`Exported ${fname}`, "success");
  }, [summary, categoryActions, pushToast]);

  /* ── derived values (all from /api/summary — row counts in hotel_reviews_preprocessed.csv) ── */
  const negTotal = summary?.negative || 0;
  const cats = summary?.categories || {};
  const sevs = summary?.severities || {};
  const highCount = sevs.High || 0;

  const formatCount = useCallback((v) => {
    if (loading) return null;
    if (summary == null) return "—";
    const n = Number(v);
    if (!Number.isFinite(n) || n < 0) return "—";
    return Math.trunc(n).toLocaleString();
  }, [loading, summary]);

  const totalReviewsLabel = formatCount(summary?.total);
  const positiveReviewsLabel = formatCount(summary?.positive);
  const negativeReviewsLabel = formatCount(summary?.negative);

  const maxSev = Math.max(sevs.High || 0, sevs.Medium || 0, sevs.Low || 0, 1);

  const filtered = useMemo(
    () =>
      reviews.filter(r => {
        if (activeTab !== "All" && r.status !== activeTab.toLowerCase()) return false;
        return true;
      }),
    [reviews, activeTab]
  );

  const selectedReview = reviews.find(r => r.id === selectedRow);

  /* ── render ── */
  return (
    <>
      <style>{CSS}</style>
      <div className="hd-root">

        {/* SIDEBAR */}
        <aside className="hd-sb">
          <div className="hd-logo">
            <div className="hd-logo-mark">
              <div className="hd-logo-icon">M</div>
              <span className="hd-logo-text">Maison Velour</span>
            </div>
            <div className="hd-logo-sub">Review Intelligence</div>
          </div>
          <nav className="hd-nav">
            <div className="hd-nl">Main</div>
            {["Dashboard", "All Reviews", "Solutions"].map((item, i) => (
              <div key={item}
                className={`hd-ni${activeNav === i ? " active" : ""}`}
                onClick={() => setActiveNav(i)}
              >
                <span className="dot" />
                {item}
                {item === "All Reviews" && (summary?.total > 0) &&
                  <span className="hd-badge">{summary.total}</span>}
              </div>
            ))}
            <div className="hd-nl" style={{ marginTop: 10 }}>Categories</div>
            {["Food", "Staff", "Rooms", "Other"].map((c, i) => (
              <div key={c}
                className={`hd-ni${activeCategory === c ? " active" : ""}`}
                onClick={() => {
                  if (activeCategory === c) {
                    setActiveCategory("All");
                  } else {
                    setActiveCategory(c);
                    setActiveNav(1);
                  }
                }}
                style={{ cursor: "pointer" }}
              >
                <span className="dot" style={{
                  background: ["#dfc06a", "#c4b5c9", "#c5a059", "#8faa96"][i]
                }} />
                {c}
                {!loading && cats[c] !== undefined &&
                  <span className="hd-num" style={{ marginLeft: "auto", fontSize: 11, color: "var(--muted)" }}>
                    {cats[c]}
                  </span>
                }
              </div>
            ))}
          </nav>
        </aside>

        {/* MAIN */}
        <main className="hd-main">
          {/* topbar */}
          <div className="hd-tb">
            <div>
              <div className="hd-tb-title">
                {activeNav === 0 && "Review Dashboard"}
                {activeNav === 1 && "All Reviews"}
                {activeNav === 2 && "AI Solutions"}
              </div>
              <div className="hd-tb-sub">
                AI-powered analysis —{" "}
                {loading
                  ? "Loading..."
                  : summary == null
                    ? "—"
                    : (
                      <>
                        <span className="hd-num">
                          {Math.max(0, Math.trunc(Number(summary.total) || 0)).toLocaleString()}
                        </span>
                        {" "}reviews processed
                      </>
                    )}
              </div>
            </div>
            <div className="hd-tb-right">
              <div className="hd-live">
                <span className="hd-live-dot" /> Live
              </div>
              <button
                className="hd-btn hd-btn-g"
                onClick={handleExportPdf}
                disabled={loading || summary == null}
                title="Download management action report (PDF)"
              >
                Export PDF
              </button>
              
              {/* Global hidden input for upload */}
              <input 
                type="file" 
                accept=".csv" 
                style={{ display: "none" }} 
                ref={fileInputRef}
                onChange={handleUpload}
              />
              <button 
                className="hd-btn hd-btn-p" 
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading || clearing || syncing}
              >
                {uploading ? "Processing..." : "Upload CSV"}
              </button>

              <button
                type="button"
                className="hd-btn hd-btn-g"
                onClick={handleSyncFromMongo}
                disabled={uploading || clearing || syncing}
                title="Load reviews from MongoDB (MONGODB_URI in api/.env), run the same ML batch as CSV upload, refresh dashboard"
              >
                {syncing ? "Syncing…" : "Sync from database"}
              </button>

              <button
                type="button"
                className="hd-btn hd-btn-g"
                style={{
                  background: "rgba(196,90,90,.1)",
                  border: "1px solid rgba(196,90,90,.35)",
                  color: "var(--red)",
                }}
                onClick={handleClearAllReviews}
                disabled={uploading || clearing || syncing}
                title="Remove all review rows and monthly batch files (models kept)"
              >
                {clearing ? "Clearing..." : "Clear all reviews"}
              </button>

              <button
                className="hd-btn hd-btn-g"
                style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)" }}
                onClick={() => window.location.reload()}
              >Refresh</button>
            </div>
          </div>

          <div className="hd-content">

            {fetchError && (
              <div style={{ background: "rgba(196,90,90,.08)", border: "1px solid rgba(196,90,90,.22)", borderRadius: 12, padding: "14px 18px", marginBottom: 20, color: "var(--red)", fontSize: 13 }}>
                Could not connect to backend: <strong>{fetchError}</strong>
                <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
                  Run the API from the <code style={{ fontSize: 11 }}>api</code> folder ({`python app.py`}), default{" "}
                  <code style={{ fontSize: 11 }}>{API.replace(/\/api$/, "")}</code>
                  . If needed set <code style={{ fontSize: 11 }}>REACT_APP_API_URL</code> in{" "}
                  <code style={{ fontSize: 11 }}>frontend/.env</code>
                  . For &quot;Sync from database&quot;, set <code style={{ fontSize: 11 }}>MONGODB_URI</code> in{" "}
                  <code style={{ fontSize: 11 }}>api/.env</code> and run <code style={{ fontSize: 11 }}>pip install pymongo</code>.
                </div>
              </div>
            )}

            {/* alert strip */}
            {showAlert && !loading && highCount > 0 && (
              <div className="hd-alert">
                <span style={{ fontSize: 13, flex: 1 }}>
                  <strong style={{ color: "var(--red)" }}>
                    <span className="hd-num">{highCount}</span>
                    {" "}high-severity complaint{highCount !== 1 ? "s" : ""}
                  </strong> require immediate attention.
                </span>
                <button className="hd-alert-x" onClick={() => setShowAlert(false)} aria-label="Dismiss">X</button>
              </div>
            )}

            {activeNav === 0 && (
              <>
                {/* METRIC CARDS */}
                <div className="hd-metrics">
                  <MetricCard label="Total reviews" color="blue" loading={loading}
                    value={totalReviewsLabel}
                    trend={summary != null && !loading ? `${summary.neg_pct}% neg` : "—"} trendDir="neu" />
                  <MetricCard label="Positive reviews" color="green" loading={loading}
                    value={positiveReviewsLabel}
                    trend={summary != null && !loading ? `${summary.pos_pct}%` : "—"} trendDir="up" />
                  <MetricCard label="Negative reviews" color="red" loading={loading}
                    value={negativeReviewsLabel}
                    trend={summary != null && !loading ? `${summary.neg_pct}%` : "—"} trendDir="down" />
                </div>

                {/* ROW 2 */}
                <div className="hd-row2">

                  {/* Category breakdown */}
                  <div className="hd-panel" style={{ animationDelay: ".25s" }}>
                    <div className="hd-ph">
                      <div className="hd-pt">
                        Complaints by category
                      </div>
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        {loading
                          ? "—"
                          : (
                            <>
                              <span className="hd-num">{negTotal}</span>
                              {" "}negative reviews
                            </>
                          )}
                      </span>
                    </div>
                    <div className="hd-pb">
                      {loading
                        ? [1, 2, 3, 4].map(i => <div key={i} className="hd-cr"><Skeleton h={8} /></div>)
                        : ["Rooms", "Staff", "Food", "Other"].map(cat => {
                          const count = cats[cat] || 0;
                          const pct = negTotal ? (count / negTotal * 100).toFixed(1) : 0;
                          const actEntry = categoryActions?.[cat];
                          const actions = actEntry?.actions || [];
                          return (
                            <div key={cat} className="hd-cat-block">
                              <div
                                className={`hd-cr hd-cr-go${activeCategory === cat ? " on" : ""}`}
                                role="button"
                                tabIndex={0}
                                title={`View ${cat} complaints`}
                                onClick={() => {
                                  setActiveCategory(cat);
                                  setActiveNav(1);
                                }}
                                onKeyDown={e => {
                                  if (e.key === "Enter" || e.key === " ") {
                                    e.preventDefault();
                                    setActiveCategory(cat);
                                    setActiveNav(1);
                                  }
                                }}
                              >
                                <span className="hd-cl" style={{ color: CAT_COLORS[cat] }}>{cat}</span>
                                <div className="hd-bt">
                                  <div className={`hd-bf ${CAT_CLS[cat]}`}
                                    style={{ width: `${pct}%` }} />
                                </div>
                                <span className="hd-cc" style={{ color: CAT_COLORS[cat] }}>{count}</span>
                                <span className="hd-cp">{pct}%</span>
                              </div>
                              <div className="hd-actions-box" onClick={e => e.stopPropagation()} role="region" aria-label={`Actions to take for ${cat}`}>
                                <div className="hd-actions-title">Actions to take</div>
                                {categoryActionsLoading ? (
                                  <>
                                    <Skeleton h={14} />
                                    <div style={{ height: 6 }} />
                                    <Skeleton w="90%" h={14} />
                                  </>
                                ) : categoryActionsError ? (
                                  <div className="hd-actions-err">Could not load suggestions.</div>
                                ) : count === 0 ? (
                                  <div className="hd-actions-empty">No complaints in this category for the selected period.</div>
                                ) : actions.length > 0 ? (
                                  <ul className="hd-actions-ul">
                                    {actions.map((line, idx) => (
                                      <li key={idx}>{line}</li>
                                    ))}
                                  </ul>
                                ) : (
                                  <div className="hd-actions-empty">No actions yet.</div>
                                )}
                              </div>
                            </div>
                          );
                        })
                      }
                    </div>
                  </div>

                  {/* Severity + donut */}
                  <div className="hd-panel" style={{ animationDelay: ".28s" }}>
                    <div className="hd-ph">
                      <div className="hd-pt">Severity breakdown</div>
                    </div>
                    <div className="hd-pb">
                      {loading
                        ? [1, 2, 3].map(i => <div key={i} className="hd-sr"><Skeleton h={30} /></div>)
                        : (
                          <>
                            {[
                              { cls: "high", label: "High", count: sevs.High || 0, color: "var(--red)", delay: ".3s" },
                              { cls: "medium", label: "Medium", count: sevs.Medium || 0, color: "var(--amber)", delay: ".4s" },
                              { cls: "low", label: "Low", count: sevs.Low || 0, color: "var(--green)", delay: ".5s" },
                            ].map(({ cls, label, count, color, delay }) => (
                              <div key={cls} className="hd-sr">
                                <span className={`hd-spill ${cls}`}>{label}</span>
                                <div className="hd-sbt">
                                  <div className={`hd-sb2 ${cls}`}
                                    style={{ width: `${(count / maxSev) * 100}%`, animation: `hd-bar .8s ${delay} ease both` }} />
                                </div>
                                <span className="hd-sn" style={{ color }}>{count}</span>
                              </div>
                            ))}
                          </>
                        )
                      }

                      <div style={{ marginTop: 22, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
                        <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 14 }}>Sentiment ratio</div>
                        {loading
                          ? <Skeleton h={90} />
                          : (() => {
                            const rawTotal = Math.max(0, Math.trunc(Number(summary?.total) || 0));
                            const posR = Math.max(0, Math.trunc(Number(summary?.positive) || 0));
                            const negR = Math.max(0, Math.trunc(Number(summary?.negative) || 0));
                            const otherR = Math.max(
                              0,
                              Math.trunc(Number(summary?.sentiment_other) || 0)
                            );
                            const circ = 2 * Math.PI * 34;
                            const denom = rawTotal > 0 ? rawTotal : 1;
                            const posDash = (posR / denom) * circ;
                            const negDash = (negR / denom) * circ;
                            const negOff = -posDash;
                            /* Percentages derived only from counts / total so the donut matches the legend */
                            const posPctDisplay = rawTotal > 0 ? (posR / rawTotal) * 100 : 0;
                            const negPctDisplay = rawTotal > 0 ? (negR / rawTotal) * 100 : 0;
                            if (rawTotal === 0) {
                              return (
                                <div style={{ fontSize: 13, color: "var(--muted)", padding: "8px 0" }}>
                                  No reviews in the dataset yet. Upload a CSV to populate metrics.
                                </div>
                              );
                            }
                            return (
                              <div className="hd-dw">
                                <svg width="90" height="90" viewBox="0 0 90 90">
                                  <circle cx="45" cy="45" r="34" fill="none" stroke="var(--surface2)" strokeWidth="10" />
                                  <circle cx="45" cy="45" r="34" fill="none" stroke="var(--green)"
                                    strokeWidth="10" strokeLinecap="round"
                                    strokeDasharray={`${posDash} ${circ - posDash}`}
                                    transform="rotate(-90 45 45)" />
                                  <circle cx="45" cy="45" r="34" fill="none" stroke="var(--red)"
                                    strokeWidth="10" strokeLinecap="round"
                                    strokeDasharray={`${negDash} ${circ - negDash}`}
                                    strokeDashoffset={negOff}
                                    transform="rotate(-90 45 45)" />
                                  <text x="45" y="42" textAnchor="middle" fill="var(--text)"
                                    fontSize="13" fontFamily="Calibri, Candara, 'Segoe UI', sans-serif" fontWeight="700">
                                    {posPctDisplay.toFixed(1)}%
                                  </text>
                                  <text x="45" y="55" textAnchor="middle" fill="var(--muted)"
                                    fontSize="8" fontFamily="DM Sans,sans-serif">positive</text>
                                </svg>
                                <div className="hd-dl">
                                  <div className="hd-dli">
                                    <span className="hd-dld" style={{ background: "var(--green)" }} />
                                    <span>Positive</span>
                                    <span className="hd-dlv" style={{ color: "var(--green)" }}>{posR.toLocaleString()}</span>
                                  </div>
                                  <div className="hd-dli">
                                    <span className="hd-dld" style={{ background: "var(--red)" }} />
                                    <span>Negative</span>
                                    <span className="hd-dlv" style={{ color: "var(--red)" }}>{negR.toLocaleString()}</span>
                                  </div>
                                  <div style={{ marginTop: 6, paddingTop: 10, borderTop: "1px solid var(--border)", fontSize: 11.5, color: "var(--muted)", lineHeight: 1.5 }}>
                                    <div>
                                      Positive:{" "}
                                      <strong className="hd-num" style={{ color: "var(--text)" }}>{posR.toLocaleString()}</strong>
                                      {" "}(<span className="hd-num">{posPctDisplay.toFixed(1)}%</span> of dataset)
                                    </div>
                                    <div>
                                      Negative:{" "}
                                      <strong className="hd-num" style={{ color: "var(--text)" }}>{negR.toLocaleString()}</strong>
                                      {" "}(<span className="hd-num">{negPctDisplay.toFixed(1)}%</span> of dataset)
                                    </div>
                                    <div style={{ marginTop: 4 }}>
                                      Total rows:{" "}
                                      <strong className="hd-num" style={{ color: "var(--text)" }}>{rawTotal.toLocaleString()}</strong>
                                      {" "}(from preprocessed CSV)
                                    </div>
                                    {otherR > 0 ? (
                                      <div style={{ marginTop: 4, color: "var(--amber)" }}>
                                        Other / unlabeled sentiment:{" "}
                                        <strong className="hd-num">{otherR.toLocaleString()}</strong>
                                        {" "}(not shown on ring)
                                      </div>
                                    ) : null}
                                  </div>
                                </div>
                              </div>
                            );
                          })()
                        }
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}

                {/* SOLUTION DETAIL (above review rows on Solutions tab) */}
                {activeNav === 2 && (
                  <>
                    {selectedReview && (
                      <div className="hd-panel" style={{ marginBottom: 24, animationDelay: ".35s" }}>
                        <div className="hd-ph">
                          <div className="hd-pt">
                            AI Solution — {selectedReview.sentiment === "Positive" ? "positive guest feedback" : `${selectedReview.category} (${selectedReview.severity} severity)`}
                          </div>
                          <div className="hd-ph-tags">
                            <span className={`hd-cb ${selectedReview.category}`}>{selectedReview.category}</span>
                            <span className={`hd-st ${selectedReview.severity === "—" ? "dash" : selectedReview.severity}`}>{selectedReview.severity === "—" ? "—" : selectedReview.severity}</span>
                          </div>
                        </div>
                        <div className="hd-pb">

                          {/* review quote */}
                          <div className="hd-quote" style={{ background: "var(--surface2)", border: "1px solid var(--border)", borderRadius: 10, padding: "14px 16px", marginBottom: 16, fontSize: 13, lineHeight: 1.6, color: "var(--muted)" }}>
                            <strong style={{ color: "var(--text)" }}>"</strong>
                            {selectedReview.review_text}
                            <strong style={{ color: "var(--text)" }}>"</strong>
                          </div>

                          {/* loading spinner */}
                          {solLoading && (
                            <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "20px 0", color: "var(--muted)" }}>
                              <span className="hd-spinner" />
                              Generating AI solution with Claude API...
                            </div>
                          )}

                          {/* error */}
                          {solError && (
                            <div className="hd-err">{solError}</div>
                          )}

                          {/* solution cards */}
                          {solution && !solLoading && (
                            <>
                              <div className="hd-sol-grid">
                                <SolCard variant="urgent" label="Immediate action (24 hrs)"
                                  text={solution.immediate_action} />
                                <SolCard variant="week" label="Short term fix (1 week)"
                                  text={solution.short_term_fix} />
                                <SolCard variant="month" label="Long term improvement (1 month)"
                                  text={solution.long_term_improvement} />
                                <SolCard variant="reply" label="Guest response message"
                                  text={`"${solution.guest_response}"`} />
                              </div>

                              <div style={{ display: "flex", gap: 12, marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--border)" }}>
                                {[
                                  { l: "Department", v: solution.department_responsible },
                                  { l: "Resolution time", v: solution.estimated_resolution_time },
                                  { l: "Prevention tip", v: solution.prevention_tip },
                                ].map(({ l, v }) => (
                                  <div key={l} className="hd-chip">
                                    <div className="hd-chip-l">{l}</div>
                                    <div className="hd-chip-v">{v}</div>
                                  </div>
                                ))}
                              </div>

                              <div style={{ display: "flex", gap: 10, marginTop: 14 }}>
                                <button className="hd-btn hd-btn-p" style={{ fontSize: 12.5 }}>
                                  Mark as resolved
                                </button>
                                <button className="hd-btn hd-btn-g" style={{ fontSize: 12.5 }}>
                                  Send response to guest
                                </button>
                                <button className="hd-btn hd-btn-g" style={{ fontSize: 12.5 }}>
                                  Forward to department
                                </button>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    )}

                    {/* prompt to select a row */}
                    {!selectedReview && !loading && reviews.length > 0 && (
                      <div className="hd-panel" style={{ marginBottom: 24, animationDelay: ".35s" }}>
                        <div className="hd-empty-hint" style={{ padding: "40px 22px", textAlign: "center", color: "var(--muted)" }}>
                          <div style={{ fontSize: 14, lineHeight: 1.55 }}>
                            Select a review in the table below to generate an AI solution.
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}

                {/* REVIEWS TABLE (All Reviews + Solutions) */}
                {(activeNav === 1 || activeNav === 2) && (
                  <div className="hd-panel" style={{ marginBottom: 24, animationDelay: ".3s" }}>
                    <div className="hd-ph">
                      <div className="hd-pt">
                        {activeCategory === "All"
                          ? "All reviews"
                          : `${activeCategory} & other feedback`} — AI solutions
                        {!reviewsLoading && (
                          <span style={{ fontSize: 12, color: "var(--muted)", fontWeight: 400 }}>
                            &nbsp;(showing <span className="hd-num">{filtered.length}</span>
                            {reviewsTotalMatching > filtered.length ? (
                              <>
                                {" "}of <span className="hd-num">{reviewsTotalMatching.toLocaleString()}</span>
                              </>
                            ) : null}
                            )
                          </span>
                        )}
                      </div>
                      <div className="hd-tabs">
                        {["All", "Pending", "Resolved"].map(t => (
                          <button key={t}
                            className={`hd-tab${activeTab === t ? " active" : ""}`}
                            onClick={() => setActiveTab(t)}>{t}</button>
                        ))}
                      </div>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                      {reviewsLoading
                        ? (
                          <div style={{ padding: "18px 22px" }}>
                            {[1, 2, 3, 4, 5].map(i => (
                              <div key={i} className="hd-skeleton hd-loading-row" />
                            ))}
                          </div>
                        )
                        : (
                          <table className="hd-tbl">
                            <thead>
                              <tr>
                                <th>Review</th>
                                <th>Category</th>
                                <th>Severity</th>
                                <th>Tags</th>
                                <th>Length</th>
                                <th>Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {filtered.map(row => (
                                <tr key={row.id}
                                  className={selectedRow === row.id ? "selected" : ""}
                                  onClick={() => handleRowClick(row)}
                                >
                                  <td><div className="hd-rv">{row.review_text}</div></td>
                                  <td>
                                    <span className={`hd-cb ${row.sentiment === "Positive" ? "Positive" : row.sentiment === "Negative" ? "Negative" : "Other"}`}>
                                      {row.sentiment || "—"}
                                    </span>
                                  </td>
                                  <td>
                                    <span className={`hd-cb ${row.category}`}>
                                      {row.category}
                                    </span>
                                  </td>
                                  <td>
                                    <span className={`hd-st ${row.severity === "—" ? "dash" : row.severity}`}>
                                      {row.severity === "—" ? "—" : row.severity}
                                    </span>
                                  </td>
                                  <td>
                                    <div className="hd-prev">{row.tags}</div>
                                  </td>
                                  <td style={{ fontSize: 12.5, color: "var(--muted)" }}>
                                    <span className="hd-num">{row.review_length}</span>
                                    {" "}words
                                  </td>
                                  <td>
                                    <span className={`hd-sd ${row.status}`}>
                                      {row.status === "pending" ? "Pending" : "Resolved"}
                                    </span>
                                  </td>
                                </tr>
                              ))}
                              {filtered.length === 0 && (
                                <tr>
                                  <td colSpan={7} style={{ textAlign: "center", color: "var(--muted)", padding: 32 }}>
                                    No reviews found
                                  </td>
                                </tr>
                              )}
                            </tbody>
                          </table>
                        )
                      }
                    </div>
                  </div>
                )}

              </div>
        </main>

        {toasts.length > 0 && (
          <div className="hd-toast-host" aria-live="polite">
            {toasts.map((t) => (
              <div key={t.id} className={`hd-toast ${t.variant}`} role="status">
                <span style={{ flex: 1 }}>{t.message}</span>
                <button
                  type="button"
                  className="hd-toast-x"
                  onClick={() => dismissToast(t.id)}
                  aria-label="Dismiss notification"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}

        {confirmDialog && (
          <div
            className="hd-modal-backdrop"
            role="presentation"
            onClick={() => !clearing && setConfirmDialog(null)}
          >
            <div
              className="hd-modal"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="hd-confirm-title"
              onClick={(e) => e.stopPropagation()}
            >
              <div id="hd-confirm-title" className="hd-modal-title">
                {confirmDialog.title || "Confirm"}
              </div>
              <div className="hd-modal-body">{confirmDialog.message}</div>
              <div className="hd-modal-actions">
                <button
                  type="button"
                  className="hd-btn hd-btn-g"
                  onClick={() => setConfirmDialog(null)}
                  disabled={clearing}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="hd-btn hd-btn-danger"
                  onClick={() => confirmDialog.onConfirm?.()}
                  disabled={clearing}
                >
                  {confirmDialog.confirmLabel || "Confirm"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}