/**
 * Makes custom components available in every .md/.mdx page without an import.
 * Usage in a page:  <Animation src="cnn/convolution" caption="..." />
 */
import MDXComponents from '@theme-original/MDXComponents';
import Animation from '@site/src/components/Animation';

export default {
  ...MDXComponents,
  Animation,
};
