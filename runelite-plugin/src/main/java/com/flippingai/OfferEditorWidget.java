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
 * Creates clickable price suggestion widgets in the GE chatbox area.
 *
 * When the GE price input is active, this injects a branded header plus
 * two clickable price lines into the chatbox container:
 *   - "▸ Market: X gp"         (gold — wiki insta price)
 *   - "▸ AI Suggest: Y gp"     (magenta — our ML prediction)
 *
 * Clicking either line auto-fills the GE price input with that value.
 */
@Slf4j
public class OfferEditorWidget
{
    // Our signature colour palette — distinct from other plugins
    private static final int COLOR_HEADER = 0xF5A623;   // Amber/gold for header
    private static final int COLOR_MARKET = 0xD4A017;   // Deep gold for market price
    private static final int COLOR_AI     = 0xE040FB;   // Magenta for AI price
    private static final int COLOR_HOVER  = 0xFFFFFF;   // White on hover
    private static final int COLOR_DIVIDER = 0x3A3A3A;  // Subtle divider line

    // Chatbox widget group and child IDs
    private static final int CHATBOX_GROUP = 162;
    private static final int CHATBOX_TITLE_CHILD = 1;
    private static final int CHATBOX_FULL_INPUT_CHILD = 5;

    private final Client client;
    private final NumberFormat nf = NumberFormat.getInstance();

    private Widget headerWidget;
    private Widget dividerWidget;
    private Widget marketWidget;
    private Widget aiWidget;

    public OfferEditorWidget(Client client, Widget chatboxContainer)
    {
        this.client = client;
        if (chatboxContainer == null)
        {
            log.debug("Chatbox container is null, cannot create price widgets");
            return;
        }

        // Layout (y offsets from top):
        //  0  — "— FlipAI —" header
        // 14  — thin divider line
        // 18  — "▸ Market: X gp"
        // 34  — "▸ AI Suggest: Y gp"
        headerWidget  = createTextWidget(chatboxContainer, 0, WidgetTextAlignment.CENTER);
        dividerWidget = createDivider(chatboxContainer, 14);
        marketWidget  = createTextWidget(chatboxContainer, 18, WidgetTextAlignment.LEFT);
        aiWidget      = createTextWidget(chatboxContainer, 34, WidgetTextAlignment.LEFT);

        // Shift existing chatbox content (title + input) down to make room
        shiftChatboxDown(52);
    }

    private Widget createTextWidget(Widget parent, int yOffset, int textAlign)
    {
        Widget widget = parent.createChild(-1, WidgetType.TEXT);
        widget.setFontId(FontID.VERDANA_11_BOLD);
        widget.setYPositionMode(WidgetPositionMode.ABSOLUTE_TOP);
        widget.setOriginalX(10);
        widget.setOriginalY(yOffset);
        widget.setOriginalHeight(16);
        widget.setXTextAlignment(textAlign);
        widget.setWidthMode(WidgetSizeMode.MINUS);
        widget.setHasListener(true);
        widget.revalidate();
        return widget;
    }

    private Widget createDivider(Widget parent, int yOffset)
    {
        Widget div = parent.createChild(-1, WidgetType.RECTANGLE);
        div.setYPositionMode(WidgetPositionMode.ABSOLUTE_TOP);
        div.setOriginalX(10);
        div.setOriginalY(yOffset);
        div.setOriginalHeight(1);
        div.setOriginalWidth(0);
        div.setWidthMode(WidgetSizeMode.MINUS);
        div.setTextColor(COLOR_DIVIDER);
        div.setOpacity(128);
        div.revalidate();
        return div;
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
        // Header
        if (headerWidget != null)
        {
            headerWidget.setText("\u2014 FlipAI \u2014");
            headerWidget.setTextColor(COLOR_HEADER);
        }

        // Market price line
        if (marketWidget != null && wikiPrice > 0)
        {
            // Show the opposite-side price as the reference boundary:
            // BUY → "Insta-Sell" (sellers' floor), SELL → "Insta-Buy" (buyers' ceiling)
            String side = isSelling ? "Insta-Buy" : "Insta-Sell";
            marketWidget.setText("\u25B8 " + side + ": " + nf.format(wikiPrice) + " gp");
            marketWidget.setTextColor(COLOR_MARKET);
            marketWidget.setAction(0, "Use price");
            addHoverListeners(marketWidget, COLOR_MARKET);
            final int price = (int) wikiPrice;
            marketWidget.setOnOpListener((JavaScriptCallback) ev -> setChatboxValue(price));
        }

        // AI price line
        if (aiWidget != null && aiPrice > 0)
        {
            aiWidget.setText("\u25B8 AI Suggest: " + nf.format(aiPrice) + " gp");
            aiWidget.setTextColor(COLOR_AI);
            aiWidget.setAction(0, "Use price");
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
        log.debug("FlipAI set GE price to {}", nf.format(value));
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
