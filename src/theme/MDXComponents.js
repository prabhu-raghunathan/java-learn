/**
 * Makes custom components available in every .md/.mdx page without an import.
 * Usage in a page:  <Animation src="cnn/convolution" caption="..." />
 *                    <Formula name="..." words="..." remember="..."> $$...$$ </Formula>
 *                    <FormulaKey />
 * Java / Spring interview components (src/components/Interview):
 *                    <PriorityKey />  <Since v="17" />
 *                    <Predict n="1" title="..." output="..."> code + explanation </Predict>
 *                    <RuleCard name="..." tag="..." why="..." remember="..."> rule </RuleCard>
 *                    <Ladder title="..."><Rung q="...">answer</Rung></Ladder>
 */
import MDXComponents from '@theme-original/MDXComponents';
import Animation from '@site/src/components/Animation';
import Formula, {FormulaKey} from '@site/src/components/Formula';
import {PriorityKey, Since, Predict, RuleCard, Ladder, Rung} from '@site/src/components/Interview';

export default {
  ...MDXComponents,
  Animation,
  Formula,
  FormulaKey,
  PriorityKey,
  Since,
  Predict,
  RuleCard,
  Ladder,
  Rung,
};
