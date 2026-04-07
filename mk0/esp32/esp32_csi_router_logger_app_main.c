/*
 * app_main.c
 *
 * Recommended usage:
 * - Start from the official `esp-csi/examples/get-started/csi_recv_router` example
 * - Replace its main/app_main.c with this file
 * - Set Wi-Fi SSID/PASSWORD in menuconfig
 * - Flash and monitor
 *
 * This code:
 * 1) connects ESP32 to one Wi-Fi router in STA mode
 * 2) enables CSI collection for packets from the connected AP
 * 3) continuously pings the gateway to generate packets
 * 4) prints CSI rows as CSV over serial
 *
 * Serial output includes a `data` column with raw CSI integers.
 * The paired Python preprocessing script can convert that CSV into .npy windows.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_idf_version.h"

#include "lwip/inet.h"
#include "lwip/ip4_addr.h"
#include "ping/ping_sock.h"

#include "protocol_examples_common.h"
#include "rom/ets_sys.h"

#define CSI_SEND_FREQUENCY_HZ 100

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const char *TAG = "csi_router_logger";

/* Print one CSV header at start. */
static bool s_header_printed = false;

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }

    /* Only keep CSI from the connected AP BSSID. */
    if (memcmp(info->mac, ctx, 6) != 0) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_seq = 0;

    if (!s_header_printed) {
        ets_printf("type,seq,mac,rssi,rate,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
        s_header_printed = true;
    }

    ets_printf(
        "CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,\"[%d",
        s_seq,
        MAC2STR(info->mac),
        rx_ctrl->rssi,
        rx_ctrl->rate,
        rx_ctrl->channel,
        rx_ctrl->timestamp,
        rx_ctrl->sig_len,
        rx_ctrl->rx_state,
        info->len,
        info->first_word_invalid,
        (int)info->buf[0]
    );

    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", (int)info->buf[i]);
    }

    ets_printf("]\"\n");
    s_seq++;
}

static void wifi_csi_init(void)
{
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = true,
        .acquire_csi_force_lltf = false,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = true,
        .acquire_csi_vht = false,
        .acquire_csi_su = false,
        .acquire_csi_mu = false,
        .acquire_csi_dcm = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg = 0,
        .dump_ack_en = false,
        .reserved = false,
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable = true,
        .acquire_csi_legacy = true,
        .acquire_csi_ht20 = true,
        .acquire_csi_ht40 = true,
        .acquire_csi_su = false,
        .acquire_csi_mu = false,
        .acquire_csi_dcm = false,
        .acquire_csi_beamformed = false,
        .acquire_csi_he_stbc = 2,
        .val_scale_cfg = false,
        .dump_ack_en = false,
        .reserved = false,
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = true,
        .manu_scale = true,
        .shift = true,
    };
#endif

    wifi_ap_record_t ap_info = {0};
    ESP_ERROR_CHECK(esp_wifi_sta_get_ap_info(&ap_info));

    ESP_LOGI(TAG, "Locked BSSID: " MACSTR ", channel=%d", MAC2STR(ap_info.bssid), ap_info.primary);

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, ap_info.bssid));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

static esp_err_t wifi_ping_router_start(void)
{
    static esp_ping_handle_t ping_handle = NULL;

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.count = 0; /* infinite */
    ping_config.interval_ms = 1000 / CSI_SEND_FREQUENCY_HZ;
    ping_config.task_stack_size = 3072;
    ping_config.data_size = 1;

    esp_netif_ip_info_t ip_info;
    esp_netif_t *sta_netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
    ESP_ERROR_CHECK(esp_netif_get_ip_info(sta_netif, &ip_info));

    ESP_LOGI(TAG, "STA IP: " IPSTR ", gateway: " IPSTR, IP2STR(&ip_info.ip), IP2STR(&ip_info.gw));

    ping_config.target_addr.type = ESP_IPADDR_TYPE_V4;
    ping_config.target_addr.u_addr.ip4.addr = ip4_addr_get_u32(&ip_info.gw);

    esp_ping_callbacks_t cbs = {0};
    ESP_ERROR_CHECK(esp_ping_new_session(&ping_config, &cbs, &ping_handle));
    ESP_ERROR_CHECK(esp_ping_start(ping_handle));

    return ESP_OK;
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /*
     * Set Wi-Fi credentials in:
     * idf.py menuconfig
     * -> Example Connection Configuration
     */
    ESP_ERROR_CHECK(example_connect());

    wifi_csi_init();
    ESP_ERROR_CHECK(wifi_ping_router_start());

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
