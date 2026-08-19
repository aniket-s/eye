import { requireElement } from '../dom.js';
import { CATEGORIES, IMAGE_FORMATS, signImagePath, type SignCategory } from '../data/categories.js';

/**
 * The browsable sign dictionary.
 *
 * Built once, lazily, the first time the page is shown.
 *
 * v1 assembled each card with `innerHTML` and inline `onerror`/`onload` attributes
 * carrying the sign name — which meant a sign containing a quote would have broken
 * out of the attribute. Building nodes with the DOM API removes that class of bug
 * entirely and is a prerequisite for the user-supplied vocabulary in Phase 5.
 */
export class DictionaryPage {
  readonly #nav = requireElement('categoryNav');
  readonly #content = requireElement('dictContent');
  readonly #modal = requireElement('modal');
  readonly #modalImg = requireElement<HTMLImageElement>('modalImg');
  readonly #modalName = requireElement('modalName');
  readonly #modalCat = requireElement('modalCat');
  #built = false;

  start(): void {
    requireElement('modalClose').addEventListener('click', () => this.#closeModal());
    this.#modal.addEventListener('click', (event) => {
      if (event.target === this.#modal) this.#closeModal();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && this.#modal.classList.contains('show')) this.#closeModal();
    });
  }

  /** Populate the page. Subsequent calls are no-ops. */
  build(): void {
    if (this.#built) return;
    this.#built = true;

    for (const category of CATEGORIES) {
      const pill = document.createElement('button');
      pill.className = 'cat-pill';
      pill.textContent = category.label;
      pill.addEventListener('click', () => {
        requireElement(`sec-${category.id}`).scrollIntoView({ behavior: 'smooth' });
      });
      this.#nav.append(pill);
    }

    for (const category of CATEGORIES) {
      this.#content.append(this.#buildSection(category));
    }
  }

  #buildSection(category: SignCategory): HTMLElement {
    const section = document.createElement('div');
    section.className = 'dict-section';
    section.id = `sec-${category.id}`;

    const title = document.createElement('div');
    title.className = 'dict-section-title';
    title.textContent = category.label;
    section.append(title);

    const grid = document.createElement('div');
    grid.className = 'signs-grid';
    for (const sign of category.signs) {
      grid.append(this.#buildCard(category, sign));
    }
    section.append(grid);

    return section;
  }

  #buildCard(category: SignCategory, sign: string): HTMLElement {
    const card = document.createElement('div');
    card.className = 'sign-card';
    card.setAttribute('role', 'button');
    card.tabIndex = 0;

    const image = document.createElement('img');
    image.alt = sign;
    const placeholder = document.createElement('div');
    placeholder.className = 'placeholder';
    placeholder.textContent = '🤟';
    placeholder.style.display = 'none';

    attachFormatFallback(image, category.id, sign, placeholder);

    const name = document.createElement('div');
    name.className = 'sign-name';
    name.textContent = sign;

    card.append(image, placeholder, name);

    const open = (): void => this.#openModal(sign, category);
    card.addEventListener('click', open);
    card.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        open();
      }
    });

    return card;
  }

  #openModal(sign: string, category: SignCategory): void {
    this.#modalImg.style.display = '';
    this.#modalImg.alt = sign;
    attachFormatFallback(this.#modalImg, category.id, sign, null);
    this.#modalName.textContent = sign;
    this.#modalCat.textContent = category.label;
    this.#modal.classList.add('show');
  }

  #closeModal(): void {
    this.#modal.classList.remove('show');
  }
}

/**
 * Try each image format in turn, revealing `placeholder` if none loads.
 *
 * The format index lives in a closure rather than on the element, so re-opening the
 * same sign always restarts from the first format.
 */
function attachFormatFallback(
  image: HTMLImageElement,
  categoryId: string,
  sign: string,
  placeholder: HTMLElement | null,
): void {
  let formatIndex = 0;

  image.onerror = (): void => {
    formatIndex += 1;
    const nextFormat = IMAGE_FORMATS[formatIndex];
    if (nextFormat !== undefined) {
      image.src = signImagePath(categoryId, sign, nextFormat);
      return;
    }
    image.style.display = 'none';
    if (placeholder !== null) placeholder.style.display = 'flex';
  };

  image.onload = (): void => {
    if (placeholder !== null) placeholder.style.display = 'none';
  };

  image.style.display = '';
  image.src = signImagePath(categoryId, sign, IMAGE_FORMATS[0]);
}
