/*
 * Copyright (c) 2009 Nokia Corporation.
 */

#include "QSingleItemSquareLayout.h"

QSingleItemSquareLayout::QSingleItemSquareLayout(QWidget* parent)
    : QLayout(parent) {
    item = 0;
    lastReceivedRect = new QRect(0, 0, 0, 0);
    _geometry = new QRect(0, 0, 0, 0);
}

QSingleItemSquareLayout::~QSingleItemSquareLayout() {
    delete item;
    delete lastReceivedRect;
    delete _geometry;
}

/* Adds item if place isn't already taken. */
void QSingleItemSquareLayout::add(QLayoutItem* item) {
    if(!hasItem()) {
        replaceItem(item);
    }
}

/* Adds item if place isn't already taken. */
void QSingleItemSquareLayout::addItem(QLayoutItem* item) {
    if(!hasItem()) {
        replaceItem(item);
    }
}

/* Adds widget if place isn't already taken. */
void QSingleItemSquareLayout::addWidget(QWidget* widget) {
    if(!hasItem()) {
        replaceItem(new QWidgetItem(widget));
    }
}

/* Returns the item pointer and dereferences it here. */
QLayoutItem* QSingleItemSquareLayout::take() {
    QLayoutItem* item = 0;
    if(this->hasItem()) {
        item = this->item;
        this->item = 0;
    }
    return item;
}

/* Returns the item pointer and dereferences it here. */
QLayoutItem* QSingleItemSquareLayout::takeAt(int index) {
    if(index != 0) {
        return 0;
    }
    return this->take();
}

/* Returns the item pointer. */
QLayoutItem* QSingleItemSquareLayout::itemAt(int index) const {
    if(index != 0) {
        return 0;
    }
    if(hasItem()) {
        return this->item;
    }
    return 0;
}

/* Checks if we have an item. */
bool QSingleItemSquareLayout::hasItem() const {
    return this->item != 0;
}

/* Returns the count of items which can be either 0 or 1. */
int QSingleItemSquareLayout::count() const {
    int returnValue = 0;
    if(hasItem()) {
        returnValue = 1;
    }
    return returnValue;
}

/* Replaces the item with the new and returns the old. */
QLayoutItem* QSingleItemSquareLayout::replaceItem(QLayoutItem* item) {
    QLayoutItem* old = 0;
    if(this->hasItem()) {
        old = this->item;
    }
    this->item = item;
    setGeometry(*this->_geometry);
    return old;
}

/* Tells which way layout expands. */
Qt::Orientations QSingleItemSquareLayout::expandingDirections() const {
    return Qt::Horizontal | Qt::Vertical;
}

/* Tells which size is preferred. */
QSize QSingleItemSquareLayout::sizeHint() const {
    return this->item->minimumSize();
}

/* Tells minimum size. */
QSize QSingleItemSquareLayout::minimumSize() const {
    return this->item->minimumSize();
}

/*
 * Tells if heightForWidth calculations is handled.
 * It isn't since width isn't enough to calculate
 * proper size.
 */
bool QSingleItemSquareLayout::hasHeightForWidth() const {
    return false;
}

/* Replaces lastReceivedRect. */
void QSingleItemSquareLayout::setLastReceivedRect(const QRect& rect) {
    QRect* oldRect = this->lastReceivedRect;
    this->lastReceivedRect = new QRect(rect.topLeft(), rect.size());
    delete oldRect;
}

/* Returns geometry */
QRect QSingleItemSquareLayout::geometry() {
    return QRect(*this->_geometry);
}

/* Sets geometry to given size. */
void QSingleItemSquareLayout::setGeometry(const QRect& rect) {
    /*
     * We check if the item is set and
     * if size is the same previously received.
     * If either is false nothing is done.
     */
    if(!this->hasItem() ||
       areRectsEqual(*this->lastReceivedRect, rect)) {
        return;
    }
    /* Replace the last received rectangle. */
    setLastReceivedRect(rect);
    /* Calculate proper size for the item relative to the received size. */
    QSize properSize = calculateProperSize(rect.size());
    /* Calculate center location in the rect and with item size. */
    QPoint properLocation = calculateCenterLocation(rect.size(), properSize);
    /* Set items geometry */
    this->item->setGeometry(QRect(properLocation, properSize));
    QRect* oldRect = this->_geometry;
    /* Cache the calculated geometry. */
    this->_geometry = new QRect(properLocation, properSize);
    delete oldRect;
    /* Super classes setGeometry */
    QLayout::setGeometry(*this->_geometry);
}

/* Takes the shortest side and creates QSize
 * with the shortest side as width and height. */
QSize QSingleItemSquareLayout::calculateProperSize(QSize from) const {
    QSize properSize;
    if(from.height() < from.width()) {
        properSize.setHeight(from.height() - this->margin());
        properSize.setWidth(from.height() - this->margin());
    }
    else {
        properSize.setWidth(from.width() - this->margin());
        properSize.setHeight(from.width() - this->margin());
    }
    return properSize;
}

/* Calculates center location from the given height and width for item size. */
QPoint QSingleItemSquareLayout::calculateCenterLocation(QSize from,
                                                        QSize itemSize) const {
    QPoint centerLocation;
    if(from.width() - from.width()/2 - itemSize.width()/2 > 0) {
        centerLocation.setX(from.width() -
                            from.width()/2 -
                            itemSize.width()/2);
    }
    if(from.height() - from.height()/2 - itemSize.height()/2 > 0) {
        centerLocation.setY(from.height() -
                            from.height()/2 -
                            itemSize.height()/2);
    }
    return centerLocation;
}

/* Compares if two QRects are equal. */
bool QSingleItemSquareLayout::areRectsEqual(const QRect& a,
                                            const QRect& b) const {
    bool result = false;
    if(a.x() == b.x() &&
       a.y() == b.y() &&
       a.height() == b.height() &&
       a.width() == b.width()) {
        result = true;
    }
    return result;
}
