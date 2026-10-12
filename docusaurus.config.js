// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// KaTeX: colour-coded symbol macros used in every AI/ML formula.
// Each wraps its argument in a CSS class (colours defined per theme in src/css/custom.css):
//   \data{x}   inputs & targets        \param{W}  learnable parameters
//   \pred{a}   outputs / activations   \loss{L}   loss & gradients
//   \hyper{η}  hyperparameters (you choose them)
const katexOptions = {
  trust: (context) => context.command === '\\htmlClass',
  strict: (errorCode) => (errorCode === 'htmlExtension' ? 'ignore' : 'warn'),
  macros: {
    '\\data': '\\htmlClass{eq-data}{#1}',
    '\\param': '\\htmlClass{eq-param}{#1}',
    '\\pred': '\\htmlClass{eq-pred}{#1}',
    '\\loss': '\\htmlClass{eq-loss}{#1}',
    '\\hyper': '\\htmlClass{eq-hyper}{#1}',
  },
};

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Java Revision Wiki',
  tagline: 'Deep-dive Java, JVM internals, multithreading, Spring Boot, databases, and system design — for interview prep.',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://prabhu-raghunathan.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'prabhu-raghunathan', // Usually your GitHub org/user name.
  projectName: 'java-learn', // Usually your repo name.

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
  },

  themes: [
    '@docusaurus/theme-mermaid',
    [
      '@easyops-cn/docusaurus-search-local',
      /** @type {import('@easyops-cn/docusaurus-search-local').PluginOptions} */
      ({
        hashed: true,
        indexDocs: true,
        indexBlog: true,
        indexPages: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
      }),
    ],
  ],

  plugins: ['docusaurus-plugin-image-zoom'],

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/prabhu-raghunathan/java-learn/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeKatex, katexOptions]],
          // custom :::analogy box (styled in src/theme/Admonition/Types.js + custom.css)
          admonitions: {keywords: ['analogy'], extendDefaults: true},
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/prabhu-raghunathan/java-learn/tree/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeKatex, katexOptions]],
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      zoom: {
        selector: '.markdown img, .docusaurus-mermaid-container svg',
        background: {
          light: 'rgb(255, 255, 255)',
          dark: 'rgb(28, 30, 33)',
        },
      },
      navbar: {
        title: 'Java Revision Wiki',
        logo: {
          alt: 'Java Revision Wiki Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Docs',
          },
          {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: 'https://github.com/prabhu-raghunathan/java-learn',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {
                label: 'Intro',
                to: '/docs/intro',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/prabhu-raghunathan/java-learn',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Prabhu Raghunathan. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        // Java (and friends) are NOT in prism-react-renderer's default bundle —
        // without this, every ```java block renders as plain uncoloured text.
        additionalLanguages: ['java', 'bash', 'json', 'yaml', 'sql', 'properties', 'groovy', 'kotlin'],
        // Line markers for code blocks (styled in src/css/custom.css):
        //   // highlight-next-line | highlight-start … highlight-end  → amber  "look here"
        //   // error-next-line     | error-start … error-end          → red    "wrong / won't compile / throws"
        //   // success-next-line   | success-start … success-end      → green  "the correct fix"
        magicComments: [
          {
            className: 'theme-code-block-highlighted-line',
            line: 'highlight-next-line',
            block: {start: 'highlight-start', end: 'highlight-end'},
          },
          {
            className: 'code-block-error-line',
            line: 'error-next-line',
            block: {start: 'error-start', end: 'error-end'},
          },
          {
            className: 'code-block-success-line',
            line: 'success-next-line',
            block: {start: 'success-start', end: 'success-end'},
          },
        ],
      },
    }),
};

export default config;
