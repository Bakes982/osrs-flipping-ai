package com.flippingai;

import lombok.extern.slf4j.Slf4j;
import net.runelite.api.Client;
import net.runelite.api.FontID;
import net.runelite.api.VarClientStr;
import net.runelite.api.widgets.JavaScriptCallback;
import net.runelite.api.widgets.Widget;
import net.runelite.api.widgets.WidgetPositionMode;
import net.runelite.api.widgets.WidgetSizeMode;
import net.runelite.api.widgets.WidgetTextAlignment;
import net.runelite.api.widgets.WidgetType;

import java.text.NumberFormat;

/**
 * Creates clickable price suggestion widgets in the GE chatbox area,
 * similar to Flipping Copilot's "set to Copilot price" feature.
 *
 * When the GE price input is active, this injects two clickable text lines
 * into the chatbox container:
 *   - "set to wiki insta buy/sell: X gp"   (cyan, clickable)
 *   - "set to AI price: Y gp"              (green, clickable)
 *
 * Clicking either line sets the GE price input to that value.
 */
@Slf4j
public class OfferEditorWidget
{
    private static final int COLOR_WIKI = 0x06B6D4;    // Cyan for wiki price
    private static final int COLOR_AI = 0x10B981;      // Green for AI price
    private static final int COLOR_HOVER = 0xFFFFFF;   // White on hover

    // Chatbox widget group and child IDs
    private static final int CHATBOX_GROUP = 162;
    private static final int CHATBOX_TITLE_CHILD = 1;
    private static final int CHATBOX_FULL_INPUT_CHILD = 5;

    private final Client client;
    private final NumberFormat nf = NumberFormat.getInstance();

    private Widget wikiWidget;
    private Widget aiWidget;

    public OfferEditorWidget(Client client, Widget chatboxContainer)
    {
        this.client = client;
        if (chatboxContainer == null)
        {
            log.debug("Chatbox container is null, cannot create price widgets");
            return;
        }

        // Create two clickable text widgets as children of the chatbox
        wikiWidget = createTextWidget(chatboxContainer, 2);
        aiWidget = createTextWidget(chatboxContainer, 18);

        // Shift existing chatbox content (title + input) down to make room
        shiftChatboxDown(35);
    }

    private Widget createTextWidget(Widget parent, int yOffset)
    {
        Widget widget = parent.createChild(-1, WidgetType.TEXT);
        widget.setFontId(FontID.VERDANA_11_BOLD);
        widget.setYPositionMode(WidgetPositionMode.ABSOLUTE_TOP);
        widget.setOriginalX(10);
        widget.setOriginalY(yOffset);
        widget.setOriginalHeight(16);
        widget.setXTextAlignment(WidgetTextAlignment.LEFT);
        widget.setWidthMode(WidgetSizeMode.MINUS);
        widget.setHasListener(true);
        widget.revalidate();
        return widget;
    }

    /**
     * Display price suggestions for the item being bought or sold.
     *
     * @param isSelling true if setting up a sell offer, false for buy
     * @param wikiPrice the wiki insta-buy (for buys) or insta-sell (for sells) price
     * @param aiPrice   the AI-suggested price from the backend
     */
    public void showPriceSuggestions(boolean isSelling, long wikiPrice, long aiPrice)
    {
        if (wikiWidget != null && wikiPrice > 0)
        {
            String label = isSelling
                ? "set to wiki insta sell: " + nf.format(wikiPrice) + " gp"
                : "set to wiki insta buy: " + nf.format(wikiPrice) + " gp";
            wikiWidget.setText(label);
            wikiWidget.setTextColor(COLOR_WIKI);
            wikiWidget.setAction(0, "Set price");
            addHoverListeners(wikiWidget, COLOR_WIKI);
            final int price = (int) wikiPrice;
            wikiWidget.setOnOpListener((JavaScriptCallback) ev -> setChatboxValue(price));
        }

        if (aiWidget != null && aiPrice > 0)
        {
            aiWidget.setText("set to AI price: " + nf.format(aiPrice) + " gp");
            aiWidget.setTextColor(COLOR_AI);
            aiWidget.setAction(0, "Set price");
            addHoverListeners(aiWidget, COLOR_AI);
            final int price = (int) aiPrice;
            aiWidget.setOnOpListener((JavaScriptCallback) ev -> setChatboxValue(price));
        }
    }

    private void addHoverListeners(Widget widget, int defaultColor)
    {
        widget.setOnMouseRepeatListener((JavaScriptCallback) ev -> widget.setTextColor(COLOR_HOVER));
        widget.setOnMouseLeaveListener((JavaScriptCallback) ev -> widget.setTextColor(defaultColor));
    }

    /**
     * Set the price in the GE chatbox input.
     * Updates both the visible text widget and the underlying VarClientStr
     * so the GE reads the correct value when the user clicks Confirm.
     */
    private void setChatboxValue(int value)
    {
        Widget chatboxInput = client.getWidget(CHATBOX_GROUP, CHATBOX_FULL_INPUT_CHILD);
        if (chatboxInput != null)
        {
            chatboxInput.setText(value + "*");
        }
        client.setVarcStrValue(VarClientStr.INPUT_TEXT, String.valueOf(value));
        log.debug("Set GE price to {}", nf.format(value));
    }

    /**
     * Move the existing chatbox title and input text down to make room
     * for our suggestion widgets at the top.
     */
    private void shiftChatboxDown(int pixels)
    {
        Widget title = client.getWidget(CHATBOX_GROUP, CHATBOX_TITLE_CHILD);
        if (title != null)
        {
            title.setOriginalY(title.getOriginalY() + pixels);
            title.revalidate();
        }

        Widget input = client.getWidget(CHATBOX_GROUP, CHATBOX_FULL_INPUT_CHILD);
        if (input != null)
        {
            input.setOriginalY(input.getOriginalY() + pixels);
            input.revalidate();
        }
    }
}
