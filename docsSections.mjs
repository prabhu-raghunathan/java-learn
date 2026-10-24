// @ts-check
// One navbar tab + one sidebar per top-level numbered folder in docs/.
// Used by sidebars.js (the sidebars) and docusaurus.config.js (the navbar tabs).
//
// To add a new subject: create docs/NN-<name>/ with a _category_.json that has a "label".
// It gets its own tab automatically (ordered by the NN prefix); nothing else to edit.

import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const DOCS_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'docs');

/** @returns {{dir: string, sidebarId: string, label: string}[]} */
export function docSections() {
  return fs
    .readdirSync(DOCS_DIR, {withFileTypes: true})
    .filter((d) => d.isDirectory() && /^\d+-/.test(d.name))
    .map((d) => d.name)
    .sort()
    .map((dir) => {
      const slug = dir.replace(/^\d+-/, '');
      let label = slug;
      try {
        const cat = JSON.parse(fs.readFileSync(path.join(DOCS_DIR, dir, '_category_.json'), 'utf8'));
        if (cat.label) label = cat.label;
      } catch {
        // no _category_.json: fall back to the folder name
      }
      // core-java → coreJavaSidebar
      const sidebarId = slug.replace(/-([a-z0-9])/g, (_, c) => c.toUpperCase()) + 'Sidebar';
      return {dir, sidebarId, label};
    });
}
