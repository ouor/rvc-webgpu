import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import tseslint from 'typescript-eslint';

// Layer rules from ARCHITECTURE.md §4, enforced with no-restricted-imports only.
const deny = (...groups) => ({ 'no-restricted-imports': ['error', { patterns: groups.map((g) => ({ group: [g], message: 'layer rule (ARCHITECTURE §4)' })) }] });

export default tseslint.config(
  { ignores: ['dist', 'external', 'docs', 'weights', 'tools', 'dev-dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: { globals: { ...globals.browser, ...globals.worker } },
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  { files: ['src/app/**'], rules: deny('**/engine/**', '**/models/**', '**/formats/**', '**/dsp/**') },
  { files: ['src/audio/**'], rules: deny('**/engine/**', '**/models/**', '**/formats/**', '**/app/**') },
  { files: ['src/engine/**', 'src/models/**'], rules: deny('**/app/**', '**/audio/**', 'react', 'react-dom') },
  { files: ['src/formats/**', 'src/dsp/**'], rules: deny('**/app/**', '**/audio/**', '**/engine/**', '**/models/**', 'onnxruntime-web*', 'react') },
  { files: ['src/shared/**'], rules: deny('**/app/**', '**/audio/**', '**/engine/**', '**/models/**', '**/formats/**', '**/dsp/**') },
);
