/**
 * Makes custom components available in every .md/.mdx page without an import.
 * Usage in a page:  <Animation src="cnn/convolution" caption="..." />
 *                    <Formula name="..." words="..." remember="..."> $$...$$ </Formula>
 *                    <FormulaKey />
 */
import MDXComponents from '@theme-original/MDXComponents';
import Animation from '@site/src/components/Animation';
import Formula, {FormulaKey} from '@site/src/components/Formula';

export default {
  ...MDXComponents,
  Animation,
  Formula,
  FormulaKey,
};
