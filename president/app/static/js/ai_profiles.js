// AI player profile registry.
// SVG functions return raw SVG strings for DOM injection in game.html.
// The designs intentionally match the AvatarSVG component in home.html.
// If you update an SVG here, update the matching case in home.html too.

(function () {
  'use strict';

  function svgVERA7(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="24,3 43,13.5 43,34.5 24,45 5,34.5 5,13.5" fill="${c}08" stroke="${c}" stroke-width="1.2" opacity="0.7"/>
      <polygon points="24,10 37,17 37,31 24,38 11,31 11,17" fill="${c}12" stroke="${c}44" stroke-width="0.8"/>
      <circle cx="24" cy="22" r="5.5" fill="${c}18" stroke="${c}" stroke-width="1.4"/>
      <circle cx="24" cy="22" r="2.5" fill="${c}" opacity="0.9"/>
      <circle cx="24" cy="22" r="1" fill="#fff" opacity="0.6"/>
      <line x1="11" y1="22" x2="18" y2="22" stroke="${c}" stroke-width="0.8" opacity="0.5"/>
      <line x1="30" y1="22" x2="37" y2="22" stroke="${c}" stroke-width="0.8" opacity="0.5"/>
      <line x1="24" y1="3" x2="24" y2="10" stroke="${c}" stroke-width="1" opacity="0.6"/>
      <circle cx="24" cy="3" r="1.5" fill="${c}" opacity="0.8"/>
      <rect x="20" y="32" width="8" height="1.5" rx="0.75" fill="${c}" opacity="0.4"/>
    </svg>`;
  }

  function svgECHO9(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="20" fill="${c}06" stroke="${c}30" stroke-width="1"/>
      <circle cx="24" cy="24" r="15" fill="${c}08" stroke="${c}50" stroke-width="1"/>
      <circle cx="24" cy="24" r="10" fill="${c}10" stroke="${c}" stroke-width="1.2"/>
      <line x1="24" y1="4" x2="24" y2="14" stroke="${c}" stroke-width="0.9" opacity="0.5"/>
      <line x1="24" y1="34" x2="24" y2="44" stroke="${c}" stroke-width="0.9" opacity="0.5"/>
      <line x1="4" y1="24" x2="14" y2="24" stroke="${c}" stroke-width="0.9" opacity="0.5"/>
      <line x1="34" y1="24" x2="44" y2="24" stroke="${c}" stroke-width="0.9" opacity="0.5"/>
      <polygon points="24,17 29,24 24,31 19,24" fill="${c}" opacity="0.85"/>
      <polygon points="24,20 27,24 24,28 21,24" fill="#fff" opacity="0.25"/>
    </svg>`;
  }

  function svgKAEL3(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M24 4 L40 12 L40 28 Q40 40 24 46 Q8 40 8 28 L8 12 Z" fill="${c}0C" stroke="${c}" stroke-width="1.3" opacity="0.8"/>
      <path d="M24 10 L35 16 L35 27 Q35 36 24 41 Q13 36 13 27 L13 16 Z" fill="${c}10" stroke="${c}44" stroke-width="0.8"/>
      <polyline points="18,30 24,16 30,30" fill="none" stroke="${c}" stroke-width="2" stroke-linejoin="round"/>
      <polyline points="20,25 24,16 28,25" fill="${c}" opacity="0.3"/>
      <rect x="18" y="30" width="12" height="2" rx="1" fill="${c}" opacity="0.7"/>
    </svg>`;
  }

  function svgAPEX1(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M8 36 L8 26 L16 32 L24 14 L32 32 L40 26 L40 36 Z" fill="${c}15" stroke="${c}" stroke-width="1.4"/>
      <path d="M10 34 L10 27 L17 32.5 L24 17 L31 32.5 L38 27 L38 34 Z" fill="${c}20"/>
      <circle cx="24" cy="14" r="2.5" fill="${c}" opacity="0.95"/>
      <circle cx="8" cy="26" r="2" fill="${c}" opacity="0.7"/>
      <circle cx="40" cy="26" r="2" fill="${c}" opacity="0.7"/>
      <rect x="8" y="34" width="32" height="3" rx="1.5" fill="${c}" opacity="0.6"/>
      <polygon points="24,19 25.5,23 29,23 26.5,25.5 27.5,29 24,27 20.5,29 21.5,25.5 19,23 22.5,23" fill="${c}" opacity="0.5"/>
    </svg>`;
  }

  function svgNULL0(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="6" y="6" width="36" height="36" rx="2" fill="${c}08" stroke="${c}40" stroke-width="1.2"/>
      <rect x="6" y="14" width="20" height="3" fill="${c}18"/>
      <rect x="6" y="28" width="14" height="2" fill="${c}10"/>
      <line x1="17" y1="20" x2="21" y2="24" stroke="${c}" stroke-width="1.8" stroke-linecap="round" opacity="0.9"/>
      <line x1="21" y1="20" x2="17" y2="24" stroke="${c}" stroke-width="1.8" stroke-linecap="round" opacity="0.9"/>
      <line x1="27" y1="20" x2="31" y2="24" stroke="${c}" stroke-width="1.8" stroke-linecap="round" opacity="0.9"/>
      <line x1="31" y1="20" x2="27" y2="24" stroke="${c}" stroke-width="1.8" stroke-linecap="round" opacity="0.9"/>
      <rect x="18" y="29" width="12" height="2" rx="1" fill="${c}50"/>
      <path d="M6,10 L6,6 L10,6" stroke="${c}" stroke-width="1.5" fill="none" stroke-linecap="square" opacity="0.6"/>
      <path d="M42,10 L42,6 L38,6" stroke="${c}" stroke-width="1.5" fill="none" stroke-linecap="square" opacity="0.6"/>
      <path d="M6,38 L6,42 L10,42" stroke="${c}" stroke-width="1.5" fill="none" stroke-linecap="square" opacity="0.6"/>
      <path d="M42,38 L42,42 L38,42" stroke="${c}" stroke-width="1.5" fill="none" stroke-linecap="square" opacity="0.6"/>
    </svg>`;
  }

  function svgWRENX(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="17,4 31,4 44,17 44,31 31,44 17,44 4,31 4,17" fill="${c}08" stroke="${c}55" stroke-width="1"/>
      <polyline points="12,12 12,20 20,20" fill="none" stroke="${c}" stroke-width="0.8" opacity="0.4"/>
      <polyline points="36,12 36,20 28,20" fill="none" stroke="${c}" stroke-width="0.8" opacity="0.4"/>
      <polyline points="12,36 12,28 20,28" fill="none" stroke="${c}" stroke-width="0.8" opacity="0.4"/>
      <polyline points="36,36 36,28 28,28" fill="none" stroke="${c}" stroke-width="0.8" opacity="0.4"/>
      <circle cx="20" cy="20" r="1.5" fill="${c}" opacity="0.6"/>
      <circle cx="28" cy="20" r="1.5" fill="${c}" opacity="0.6"/>
      <circle cx="20" cy="28" r="1.5" fill="${c}" opacity="0.6"/>
      <circle cx="28" cy="28" r="1.5" fill="${c}" opacity="0.6"/>
      <circle cx="24" cy="24" r="5" fill="${c}15" stroke="${c}" stroke-width="1.2"/>
      <line x1="24" y1="19" x2="24" y2="29" stroke="${c}" stroke-width="0.8" opacity="0.7"/>
      <line x1="19" y1="24" x2="29" y2="24" stroke="${c}" stroke-width="0.8" opacity="0.7"/>
      <circle cx="24" cy="24" r="2" fill="${c}" opacity="0.85"/>
    </svg>`;
  }

  function svgDUSK5(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="24,2 28,18 44,14 34,26 46,34 30,32 32,46 24,36 16,46 18,32 2,34 14,26 4,14 20,18" fill="${c}10" stroke="${c}" stroke-width="1" opacity="0.7"/>
      <circle cx="24" cy="24" r="9" fill="${c}15" stroke="${c}" stroke-width="1.4"/>
      <circle cx="19" cy="19" r="1.6" fill="${c}" opacity="0.9"/>
      <circle cx="29" cy="19" r="1.6" fill="${c}" opacity="0.9"/>
      <circle cx="24" cy="24" r="1.6" fill="${c}" opacity="0.9"/>
      <circle cx="19" cy="29" r="1.6" fill="${c}" opacity="0.9"/>
      <circle cx="29" cy="29" r="1.6" fill="${c}" opacity="0.9"/>
    </svg>`;
  }

  function svgRHEA2(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="24,3 45,24 24,45 3,24" fill="${c}08" stroke="${c}" stroke-width="1.2" opacity="0.7"/>
      <polygon points="24,10 38,24 24,38 10,24" fill="${c}10" stroke="${c}50" stroke-width="0.9"/>
      <line x1="24" y1="10" x2="24" y2="38" stroke="${c}" stroke-width="0.6" opacity="0.25"/>
      <line x1="10" y1="24" x2="38" y2="24" stroke="${c}" stroke-width="0.6" opacity="0.25"/>
      <line x1="14" y1="14" x2="34" y2="34" stroke="${c}" stroke-width="0.6" opacity="0.18"/>
      <line x1="34" y1="14" x2="14" y2="34" stroke="${c}" stroke-width="0.6" opacity="0.18"/>
      <rect x="18" y="29" width="3" height="6" fill="${c}" opacity="0.5"/>
      <rect x="22.5" y="23" width="3" height="12" fill="${c}" opacity="0.75"/>
      <rect x="27" y="26" width="3" height="9" fill="${c}" opacity="0.6"/>
      <circle cx="24" cy="3" r="1.8" fill="${c}" opacity="0.8"/>
    </svg>`;
  }

  function svgGRUB4(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="24" cy="26" rx="17" ry="15" fill="${c}10" stroke="${c}" stroke-width="1.2" opacity="0.7"/>
      <ellipse cx="24" cy="26" rx="12" ry="10" fill="${c}14" stroke="${c}44" stroke-width="0.8"/>
      <ellipse cx="24" cy="34" rx="7" ry="2.5" fill="${c}30" stroke="${c}" stroke-width="1"/>
      <ellipse cx="24" cy="31.5" rx="7" ry="2.5" fill="${c}20" stroke="${c}" stroke-width="1"/>
      <ellipse cx="24" cy="29" rx="7" ry="2.5" fill="${c}30" stroke="${c}" stroke-width="1"/>
      <line x1="21" y1="29" x2="27" y2="29" stroke="${c}88" stroke-width="0.8"/>
      <circle cx="19" cy="21" r="3.5" fill="${c}18" stroke="${c}" stroke-width="1.2"/>
      <circle cx="29" cy="21" r="3.5" fill="${c}18" stroke="${c}" stroke-width="1.2"/>
      <circle cx="19" cy="21" r="1.5" fill="${c}" opacity="0.9"/>
      <circle cx="29" cy="21" r="1.5" fill="${c}" opacity="0.9"/>
      <text x="24" y="16" text-anchor="middle" font-family="monospace" font-size="7" fill="${c}" opacity="0.7" font-weight="bold">$</text>
      <path d="M7 28 Q4 24 7 20 Q10 22 8 26" fill="${c}20" stroke="${c}" stroke-width="0.9"/>
      <path d="M41 28 Q44 24 41 20 Q38 22 40 26" fill="${c}20" stroke="${c}" stroke-width="0.9"/>
    </svg>`;
  }

  function svgFUZZ8(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="24" cy="25" rx="16" ry="17" fill="${c}0C" stroke="${c}" stroke-width="1.2" opacity="0.7"/>
      <circle cx="18" cy="23" r="4" fill="${c}10" stroke="${c}" stroke-width="1.1"/>
      <circle cx="30" cy="23" r="4" fill="${c}10" stroke="${c}" stroke-width="1.1"/>
      <circle cx="18" cy="23" r="1.2" fill="${c}" opacity="0.4"/>
      <circle cx="30" cy="23" r="1.2" fill="${c}" opacity="0.4"/>
      <path d="M17 33 Q20 30 24 33 Q28 36 31 33" fill="none" stroke="${c}" stroke-width="1.2" stroke-linecap="round"/>
      <text x="10" y="14" font-family="monospace" font-size="9" fill="${c}" opacity="0.55" font-weight="bold">?</text>
      <text x="33" y="11" font-family="monospace" font-size="7" fill="${c}" opacity="0.35">?</text>
      <text x="22" y="9" font-family="monospace" font-size="6" fill="${c}" opacity="0.25">?</text>
      <path d="M18 8 Q22 4 26 8" fill="none" stroke="${c}" stroke-width="1" stroke-linecap="round" opacity="0.5"/>
      <path d="M20 6 Q24 2 28 6" fill="none" stroke="${c}" stroke-width="0.8" stroke-linecap="round" opacity="0.3"/>
      <circle cx="24" cy="8" r="1.5" fill="${c}" opacity="0.6"/>
    </svg>`;
  }

  function svgSLAB6(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="10" y="10" width="28" height="28" rx="4" fill="${c}0C" stroke="${c}" stroke-width="1.2" opacity="0.7" transform="rotate(8 24 24)"/>
      <rect x="14" y="14" width="20" height="20" rx="3" fill="${c}10" stroke="${c}40" stroke-width="0.8" transform="rotate(-4 24 24)"/>
      <ellipse cx="19" cy="21" rx="3.5" ry="2" fill="${c}20" stroke="${c}" stroke-width="1.1"/>
      <ellipse cx="29" cy="22" rx="3.5" ry="2" fill="${c}20" stroke="${c}" stroke-width="1.1"/>
      <circle cx="18" cy="21" r="1.2" fill="${c}" opacity="0.8"/>
      <circle cx="30" cy="22" r="1.2" fill="${c}" opacity="0.8"/>
      <path d="M18 31 L22 30 L28 32" fill="none" stroke="${c}" stroke-width="1.3" stroke-linecap="round"/>
      <rect x="5" y="8" width="4" height="4" rx="1" fill="${c}30" stroke="${c}" stroke-width="0.8" transform="rotate(20 7 10)"/>
      <rect x="39" y="36" width="3" height="3" rx="1" fill="${c}25" stroke="${c}" stroke-width="0.8" transform="rotate(-15 40 37)"/>
      <rect x="38" y="9" width="5" height="2" rx="1" fill="${c}20" stroke="${c}" stroke-width="0.8" transform="rotate(35 40 10)"/>
      <rect x="5" y="37" width="3" height="5" rx="1" fill="${c}28" stroke="${c}" stroke-width="0.8" transform="rotate(-25 6 39)"/>
    </svg>`;
  }

  function svgZIPZ0(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="26" r="16" fill="${c}0C" stroke="${c}" stroke-width="1.2" opacity="0.7"/>
      <circle cx="18" cy="23" r="4" fill="${c}12" stroke="${c}" stroke-width="1.1"/>
      <circle cx="30" cy="23" r="4" fill="${c}12" stroke="${c}" stroke-width="1.1"/>
      <circle cx="16" cy="21" r="1.5" fill="${c}" opacity="0.85"/>
      <circle cx="32" cy="25" r="1.5" fill="${c}" opacity="0.85"/>
      <ellipse cx="24" cy="33" rx="4" ry="3" fill="${c}18" stroke="${c}" stroke-width="1.1"/>
      <line x1="24" y1="10" x2="24" y2="4" stroke="${c}" stroke-width="1" opacity="0.5"/>
      <polygon points="24,2 22,6 26,6" fill="${c}" opacity="0.5"/>
      <line x1="37" y1="16" x2="42" y2="11" stroke="${c}" stroke-width="1" opacity="0.4"/>
      <polygon points="44,9 40,11 42,15" fill="${c}" opacity="0.4"/>
      <line x1="11" y1="16" x2="6" y2="11" stroke="${c}" stroke-width="1" opacity="0.35"/>
      <polygon points="4,9 8,11 6,15" fill="${c}" opacity="0.35"/>
      <line x1="40" y1="30" x2="45" y2="33" stroke="${c}" stroke-width="1" opacity="0.3"/>
      <polygon points="47,35 43,32 42,36" fill="${c}" opacity="0.3"/>
      <circle cx="24" cy="10" r="2.5" fill="${c}20" stroke="${c}" stroke-width="1" opacity="0.7"/>
      <line x1="24" y1="7.5" x2="24" y2="12.5" stroke="${c}" stroke-width="0.8" opacity="0.6"/>
      <line x1="21.5" y1="10" x2="26.5" y2="10" stroke="${c}" stroke-width="0.8" opacity="0.6"/>
    </svg>`;
  }

  function svgDefault(c, s) {
    return `<svg width="${s}" height="${s}" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="24" cy="24" r="20" fill="${c}10" stroke="${c}" stroke-width="1.2"/>
      <text x="24" y="29" text-anchor="middle" font-family="monospace" font-size="14" fill="${c}" font-weight="600">AI</text>
    </svg>`;
  }

  window.AI_PROFILES = {
    'VERA-7': { label: 'Analyst',   difficulty: 'Hard',   svgFn: svgVERA7   },
    'KAEL-3': { label: 'Aggressor', difficulty: 'Hard',   svgFn: svgKAEL3   },
    'APEX-1': { label: 'Sovereign', difficulty: 'Hard',   svgFn: svgAPEX1   },
    'RHEA-2': { label: 'Tactician', difficulty: 'Hard',   svgFn: svgRHEA2   },
    'NULL-0': { label: 'Phantom',   difficulty: 'Medium', svgFn: svgNULL0   },
    'WREN-X': { label: 'Schemer',   difficulty: 'Medium', svgFn: svgWRENX   },
    'DUSK-5': { label: 'Rogue',     difficulty: 'Medium', svgFn: svgDUSK5   },
    'ECHO-9': { label: 'Mimic',     difficulty: 'Medium', svgFn: svgECHO9   },
    'GRUB-4': { label: 'Greedy',    difficulty: 'Easy',   svgFn: svgGRUB4   },
    'FUZZ-8': { label: 'Forgetful', difficulty: 'Easy',   svgFn: svgFUZZ8   },
    'SLAB-6': { label: 'Careless',  difficulty: 'Easy',   svgFn: svgSLAB6   },
    'ZIPZ-0': { label: 'Distracted',difficulty: 'Easy',   svgFn: svgZIPZ0   },
  };

  // Returns an SVG HTML string for a given player name.
  // playerName may be "VERA-7 (AI)", "VERA-7", or any human name.
  window.getAvatarSVG = function (playerName, color, size) {
    const id      = (playerName || '').replace(/ \(AI\)$/, '').trim();
    const c       = color || '#00FFB8';
    const s       = size  || 36;
    const profile = window.AI_PROFILES[id];
    return profile ? profile.svgFn(c, s) : svgDefault(c, s);
  };
}());
