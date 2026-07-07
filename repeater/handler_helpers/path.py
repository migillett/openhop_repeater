import logging
import time

logger = logging.getLogger("PathHelper")


class PathHelper:
    def __init__(self, acl_dict=None, log_fn=None, ack_received_fn=None):

        self.acl_dict = acl_dict or {}
        self.log_fn = log_fn or logger.info
        # Async callback fed with ACK CRCs found embedded in PATH payloads
        # (dispatcher._register_ack_received) so local waiters resolve.
        self.ack_received_fn = ack_received_fn

    async def process_path_packet(self, packet):

        from openhop_core.protocol.crypto import CryptoUtils

        try:
            if len(packet.payload) < 2:
                return False

            dest_hash = packet.payload[0]
            src_hash = packet.payload[1]

            # Get the ACL for this destination identity
            identity_acl = self.acl_dict.get(dest_hash)
            if not identity_acl:
                logger.debug(f"No ACL for dest 0x{dest_hash:02X}, allowing forward")
                return False

            # Find the client by source hash
            client = None
            for client_info in identity_acl.get_all_clients():
                pubkey = client_info.id.get_public_key()
                if pubkey[0] == src_hash:
                    client = client_info
                    break

            if not client:
                logger.debug(f"PATH packet from unknown client 0x{src_hash:02X}, allowing forward")
                return False

            # Get shared secret for decryption
            shared_secret = client.shared_secret
            if not shared_secret or len(shared_secret) == 0:
                logger.debug(f"No shared secret for client 0x{src_hash:02X}, cannot decrypt PATH")
                return False

            # Decrypt the PATH packet payload
            # Payload format: dest_hash(1) + src_hash(1) + mac(2) + encrypted_data
            if len(packet.payload) < 4:
                logger.debug(f"PATH packet too short: {len(packet.payload)} bytes")
                return False

            mac_and_data = packet.payload[2:]  # Skip dest_hash and src_hash
            aes_key = shared_secret[:16]
            decrypted = CryptoUtils.mac_then_decrypt(aes_key, shared_secret, mac_and_data)

            if not decrypted:
                logger.debug(f"Failed to decrypt PATH packet from 0x{src_hash:02X}")
                return False

            # Parse decrypted PATH data
            # Format: path_len(1) + path[...] + extra_type(1) + extra[...]
            # path_len is the ENCODED wire byte (bits 0-5 = hash count, bits
            # 6-7 = hash size - 1), matching Packet.path_len — with 3-byte
            # hashes an empty path is 0x80, not 0x00. Reading it as a raw byte
            # count made every such path return look truncated.
            if len(decrypted) < 1:
                logger.debug("Decrypted PATH data too short")
                return False

            from openhop_core.protocol.packet_utils import PathUtils

            path_len_byte = decrypted[0]
            if not PathUtils.is_valid_path_len(path_len_byte):
                logger.debug(f"Invalid encoded path_len 0x{path_len_byte:02X} in PATH data")
                return False
            path_byte_len = PathUtils.get_path_byte_len(path_len_byte)
            if len(decrypted) < 1 + path_byte_len:
                logger.debug(
                    f"PATH data truncated: need {1 + path_byte_len} bytes, got {len(decrypted)}"
                )
                return False

            path_data = decrypted[1 : 1 + path_byte_len]

            # Update client's out_path (same as C++ memcpy); out_path_len keeps
            # the encoded byte so direct sends put it on the wire as-is.
            client.out_path = bytearray(path_data)
            client.out_path_len = path_len_byte
            client.last_activity = int(time.time())

            logger.info(
                f"Updated out_path for client 0x{src_hash:02X} -> 0x{dest_hash:02X}: "
                f"path_len=0x{path_len_byte:02X}, path={[hex(b) for b in path_data]}"
            )

            # Extra section after the path: extra_type(1) + extra[...]. Firmware
            # answers a flood-received DM with a path return that embeds the
            # delivery ACK here (createPathReturn, extra_type=PAYLOAD_TYPE_ACK);
            # register it so local waiters (e.g. room server pushes) resolve.
            from openhop_core.protocol.constants import PAYLOAD_TYPE_ACK

            extra_start = 1 + path_byte_len
            if (
                self.ack_received_fn is not None
                and len(decrypted) >= extra_start + 5
                and decrypted[extra_start] == PAYLOAD_TYPE_ACK
            ):
                ack_crc = int.from_bytes(
                    bytes(decrypted[extra_start + 1 : extra_start + 5]), "little"
                )
                await self.ack_received_fn(ack_crc)
                logger.info(
                    f"PATH from 0x{src_hash:02X} carried embedded ACK CRC={ack_crc:08X}"
                )

            # Don't mark as do_not_retransmit - let it forward normally
            return False

        except Exception as e:
            logger.error(f"Error processing PATH packet: {e}", exc_info=True)
            return False
