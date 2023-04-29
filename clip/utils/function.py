from typing import Generator, Iterable, Iterator, Optional, Tuple, TypeVar

Item = TypeVar("Item")
CLIPItem = TypeVar("CLIPItem")
ImageItem = TypeVar("ImageItem")
TextItem = TypeVar("TextItem")


def get_next(iterator: Iterator[Item]) -> Optional[Item]:
    try:
        return next(iterator)
    except StopIteration:
        return None


def zip_dataloaders(
    clip: Iterable[CLIPItem], image: Iterable[ImageItem], text: Iterable[TextItem]
) -> Generator[Tuple[CLIPItem, Optional[ImageItem], Optional[TextItem]], None, None]:
    clip_itertor = iter(clip)
    image_itertor = iter(image)
    text_itertor = iter(text)
    clip_current = get_next(clip_itertor)
    image_current = get_next(image_itertor)
    text_current = get_next(text_itertor)
    while clip_current is not None:
        yield clip_current, image_current, text_current
        clip_current = get_next(clip_itertor)
        if image_current is not None:
            image_current = get_next(image_itertor)
        if text_current is not None:
            text_current = get_next(text_itertor)
